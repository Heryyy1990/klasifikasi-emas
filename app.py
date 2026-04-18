import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Arsip Pintar", page_icon="🗂️", layout="centered")

# --- 1. LOGIKA RULE-BASED (Kata Kunci Pasti) ---
def manual_rule(text):
    text = text.lower()
    # Anda bisa menambah aturan kata kunci di sini
    if re.search(r'(pangkat|mutasi|nip|pns|asn|cpns|pegawai|cuti)', text):
        return "800", "Kepegawaian"
    if re.search(r'(sppd|perjalanan dinas|tugas luar)', text):
        return "090", "Perjalanan Dinas"
    if re.search(r'(undangan|rapat|pertemuan)', text):
        return "005", "Undangan"
    return None, None

# --- 2. LOGIKA NLP/ML (Pencarian Kemiripan Makna) ---
def nlp_classification(text, df, top_n=3):
    vectorizer = TfidfVectorizer()
    # Menggabungkan semua 'uraian' di database dengan input user di posisi terakhir
    semua_teks = df['uraian'].fillna("").tolist() + [text]
    tfidf_matrix = vectorizer.fit_transform(semua_teks)
    
    # Menghitung kemiripan (Cosine Similarity)
    kemiripan = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # Mengambil indeks dengan skor tertinggi
    indeks_teratas = kemiripan.argsort()[-top_n:][::-1]
    
    hasil = []
    for idx in indeks_teratas:
        if kemiripan[idx] > 0: # Pastikan ada kemiripan walau sedikit
            hasil.append({
                'kode': df.iloc[idx]['kode'],
                'uraian': df.iloc[idx]['uraian'],
                'skor': kemiripan[idx]
            })
    return hasil

# --- 3. MEMUAT DATABASE (Standar Emas) ---
@st.cache_data
def load_data():
    try:
        # Langsung tembak menggunakan pemisah koma (,). 
        # on_bad_lines='skip' akan membuang baris yang rusak agar aplikasi tidak error
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip')
    except:
        # Jika masih gagal, coba dengan titik koma
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip')
    
    # Bersihkan judul kolom (berjaga-jaga jika ada spasi gaib)
    df.columns = df.columns.str.strip().str.lower().str.replace('"', '').str.replace("'", "")
    
    # Jika datanya masih keras kepala menyatu di satu kolom
    if 'uraian' not in df.columns and len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
    
    # Pastikan tidak ada data yang kosong (NaN) agar AI tidak bingung
    if 'uraian' in df.columns:
        df['uraian'] = df['uraian'].fillna("")
    if 'kode' in df.columns:
        df['kode'] = df['kode'].fillna("000")
        
    return df

# --- 4. ANTARMUKA (UI) STREAMLIT ---
st.title("🗂️ Asisten Klasifikasi Arsip Pintar")
st.markdown("**Berdasarkan Permendagri No. 83 Tahun 2022**")

try:
    # Load database
    df = load_data()
    
    # Kolom input untuk user
    user_input = st.text_input(
        "Ketik Perihal / Isi Ringkas Surat:", 
        placeholder="Contoh: Undangan Rapat Evaluasi TAPD..."
    )

    if user_input:
        st.write("---")
        st.subheader("Hasil Analisis Sistem:")
        
        # 1. Cek dengan Rule-Based terlebih dahulu
        kode_rule, uraian_rule = manual_rule(user_input)
        
        if kode_rule:
            st.success("Tangkapan Kata Kunci (Rule-Based) Ditemukan!")
            st.info(f"Rekomendasi Cepat: **{kode_rule}** - {uraian_rule}")
            st.write("*Berikut adalah opsi klasifikasi lebih spesifik dari AI:*")
            
        # 2. Jalankan NLP untuk mencari sub-kode yang paling pas
        hasil_nlp = nlp_classification(user_input, df)
        
        if hasil_nlp:
            for i, res in enumerate(hasil_nlp):
                # Menampilkan hasil (Buka otomatis untuk ranking 1)
                with st.expander(f"Rekomendasi #{i+1}: Kode {res['kode']} (Akurasi: {res['skor']:.1%})", expanded=(i==0)):
                    st.write(f"**Uraian:** {res['uraian']}")
        else:
            st.warning("Sistem tidak menemukan klasifikasi yang mirip di database. Coba gunakan kata kunci yang lebih baku.")

except FileNotFoundError:
    st.error("❌ File 'klasifikasi_arsip_emas.csv' tidak ditemukan. Pastikan Anda sudah mengunggahnya ke repository GitHub Anda.")
