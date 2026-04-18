import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Klasifikasi Arsip Pintar", page_icon="🗂️", layout="centered")

# --- 1. MEMUAT DATABASE (Standar Emas) ---
@st.cache_data
def load_data():
    # Membaca file dengan aman
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip', dtype={'kode': str})
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip', dtype={'kode': str})
    
    df.columns = df.columns.str.strip().str.lower().str.replace('"', '').str.replace("'", "")
    
    if 'uraian' not in df.columns and len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
    
    # Pastikan bersih dan berformat string
    df['uraian'] = df['uraian'].fillna("").astype(str)
    df['kode'] = df['kode'].fillna("000").astype(str).str.strip()
    
    return df

# --- 2. FITUR HIERARKI KODE ---
def get_hierarchy(kode_target, df):
    parts = str(kode_target).split('.')
    hierarchy_list = []
    current_code = ""
    tingkatan = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier", "Sektier"]

    for i, part in enumerate(parts):
        # Merangkai kode dari level teratas ke bawah
        if i == 0:
            current_code = part
        else:
            current_code += "." + part
        
        # Mencari uraiannya di database
        match = df[df['kode'] == current_code]
        if not match.empty:
            uraian = match.iloc[0]['uraian'].title()
        else:
            uraian = "Menyesuaikan Induk"
            
        label = tingkatan[i] if i < len(tingkatan) else f"Level {i+1}"
        hierarchy_list.append(f"└─ **{current_code}**: {uraian} *({label})*")
        
    return hierarchy_list

# --- 3. LOGIKA RULE-BASED (Akurasi Pasti) ---
def manual_rule(text):
    text = text.lower()
    if re.search(r'(pangkat|mutasi|nip|pns|asn|cpns|pegawai|cuti)', text):
        return "800.1" # Diarahkan ke rumpun Kepegawaian yg lebih spesifik
    if re.search(r'(sppd|perjalanan dinas|tugas luar)', text):
        return "090"
    if re.search(r'(undangan|rapat|pertemuan)', text):
        return "005"
    return None

# --- 4. LOGIKA NLP/ML (Akurasi Tinggi dengan N-Gram) ---
def nlp_classification(text, df, top_n=3):
    # Menggunakan N-Gram (1,2) agar AI paham frasa kata
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    semua_teks = df['uraian'].tolist() + [text]
    
    tfidf_matrix = vectorizer.fit_transform(semua_teks)
    kemiripan = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    indeks_teratas = kemiripan.argsort()[-top_n:][::-1]
    
    hasil = []
    for idx in indeks_teratas:
        if kemiripan[idx] > 0.05: # Batas minimal kemiripan agar tidak asal tebak
            hasil.append({
                'kode': df.iloc[idx]['kode'],
                'uraian': df.iloc[idx]['uraian'],
                'skor': kemiripan[idx]
            })
    return hasil

# --- 5. ANTARMUKA (UI) STREAMLIT ---
st.title("🗂️ SIKAP - Sistem Klasifikasi Arsip Pintar")
st.markdown("**Berdasarkan Permendagri No. 83 Tahun 2022**")

try:
    df = load_data()
    
    user_input = st.text_input(
        "Ketik Perihal / Isi Ringkas Surat:", 
        placeholder="Contoh: Undangan Rapat Evaluasi Anggaran..."
    )

    if user_input:
        st.write("---")
        st.subheader("💡 Hasil Analisis Klasifikasi:")
        
        # Cek Rule-Based
        kode_rule = manual_rule(user_input)
        if kode_rule:
            st.success("📌 Ditemukan Kecocokan Pola Khusus (Rule-Based)")
            st.write("Jalur Hierarki:")
            hierarki_rule = get_hierarchy(kode_rule, df)
            for h in hierarki_rule:
                st.markdown(h)
            st.write("---")
            st.write("*Opsi Klasifikasi Spesifik dari AI:*")
            
        # Cek NLP
        hasil_nlp = nlp_classification(user_input, df)
        
        if hasil_nlp:
            for i, res in enumerate(hasil_nlp):
                with st.expander(f"🏆 Rekomendasi #{i+1}: Kode {res['kode']} (Kemiripan: {res['skor']:.1%})", expanded=(i==0)):
                    st.write("**Jalur Hierarki Kode Ini:**")
                    # Memanggil fitur hierarki untuk hasil AI
                    hierarki_ai = get_hierarchy(res['kode'], df)
                    for h in hierarki_ai:
                        st.markdown(h)
        else:
            st.warning("⚠️ Sistem tidak menemukan klasifikasi yang mirip. Coba gunakan kata kunci dinas yang lebih baku.")

except Exception as e:
    st.error(f"Terjadi kesalahan sistem. Pastikan file CSV sudah benar. Log: {e}")
