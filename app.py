import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import fuzz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP Dashboard", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- INISIALISASI NLP ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

# --- KAMUS JARGON BIROKRASI (SUPER LENGKAP) ---
kamus_birokrasi = {
    # MASUKKAN SELURUH ISI KAMUS PANJANG ANDA DI SINI
    "apbd": "anggaran pendapatan dan belanja daerah",
    "tapd": "tim anggaran pemerintah daerah",
    "sppd": "surat perintah perjalanan dinas",
    "lhe": "laporan hasil evaluasi",
    "sp2d": "surat perintah pencairan dana",
    "dak": "dana alokasi khusus",
    "pns": "pegawai negeri sipil",
    "asn": "aparatur sipil negara",
    "dpa": "dokumen pelaksanaan anggaran",
    "spp": "surat permintaan pembayaran",
    "spm": "surat perintah membayar",
    "ls": "langsung"
    # Dst... pastikan isi lengkapnya Anda paste kembali ke sini ya!
}

def terjemahkan_singkatan(text):
    kata_kata = str(text).lower().split()
    kata_terjemahan = [kamus_birokrasi.get(kata, kata) for kata in kata_kata]
    return " ".join(kata_terjemahan)

def preprocess_text(text):
    text = str(text).lower()
    text = terjemahkan_singkatan(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = remover.remove(text)
    text = stemmer.stem(text)
    return text

# --- MANAJEMEN DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', dtype=str)
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', dtype=str)
    
    df.columns = ['kode', 'uraian'] + list(df.columns[2:])
    df['uraian'] = df['uraian'].fillna("").astype(str)
    df['clean_uraian'] = df['uraian'].apply(preprocess_text)
    return df

def save_feedback(input_text, final_code):
    file_path = 'log_penggunaan.csv'
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[now, input_text, final_code]], columns=['Waktu', 'Input', 'Kode_Dipilih'])
    
    if not os.path.isfile(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)

# --- OTAK PENCARIAN (VERSI PRESISI BEBAS GEMBOK 100% ASLI) ---
def smart_classify(user_input, df, top_n=3):
    clean_input = preprocess_text(user_input)
    
    # N-Gram 1 sampai 3 kata sekaligus
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    final_scores = []
    for idx, score in enumerate(cosine_sim):
        # Menggunakan token_sort_ratio ketat & Bobot 85% / 15%
        fuzzy_score = fuzz.token_sort_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        combined_score = (score * 0.85) + (fuzzy_score * 0.15)
        final_scores.append((idx, combined_score))
        
    hasil_akhir = sorted(final_scores, key=lambda x: x[1], reverse=True)
    return hasil_akhir[:top_n]


# --- UI UTAMA / DASHBOARD ---
st.sidebar.title("🏢 SIKAP Muna Barat")
st.sidebar.markdown("Sistem Informasi Klasifikasi Arsip Pintar")
menu = st.sidebar.radio("Navigasi Menu:", ["🔍 Pencarian Pintar", "📖 Kamus Kode", "📊 Laporan Strategis"])

df = load_data()

if menu == "🔍 Pencarian Pintar":
    st.title("🔍 Pencarian Klasifikasi")
    user_input = st.text_input("Ketik Perihal Surat:", placeholder="Contoh: laporan hasil evaluasi dana alokasi khusus...")

    if user_input:
        with st.spinner("Menganalisis makna persuratan..."):
            # MEMANGGIL OTAK "BEBAS GEMBOK" DI SINI
            results = smart_classify(user_input, df)
            
            st.subheader("Rekomendasi Terbaik:")
            for i, (idx, score) in enumerate(results):
                res = df.iloc[idx]
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>🏆 #{i+1} | Kode: {res['kode']} (Skor: {score:.1%})</h4>
                        <p><b>Uraian:</b> {res['uraian']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"✅ Gunakan Kode {res['kode']}", key=f"btn_{idx}"):
                        save_feedback(user_input, res['kode'])
                        st.success(f"Berhasil! Kode {res['kode']} direkam untuk Laporan Pimpinan.")

elif menu == "📖 Kamus Kode":
    st.title("📖 Penelusuran Kamus Kode")
    search_code = st.text_input("Masukkan Kode (Misal: 800):")
    
    if search_code:
        filtered_df = df[df['kode'].str.startswith(search_code)]
        if not filtered_df.empty:
            st.write(f"Menampilkan **{len(filtered_df)}** uraian untuk rumpun kode **{search_code}**:")
            st.table(filtered_df[['kode', 'uraian']])
        else:
            st.warning("Kode tidak ditemukan dalam database.")

elif menu == "📊 Laporan Strategis":
    st.title("📊 Laporan Tren Arsip")
    if os.path.isfile('log_penggunaan.csv'):
        log_df = pd.read_csv('log_penggunaan.csv')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Arsip Diklasifikasikan", len(log_df))
        
        st.subheader("Top 5 Klasifikasi Paling Sering Muncul")
        top_codes = log_df['Kode_Dipilih'].value_counts().head(5)
        st.bar_chart(top_codes)
        
        st.subheader("Riwayat Aktivitas Arsiparis")
        st.dataframe(log_df.tail(10))
    else:
        st.info("Belum ada data laporan. Konfirmasi pencarian kode di menu utama terlebih dahulu.")
