import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import process, fuzz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️")

# --- INISIALISASI NLP (Sastrawi) ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

def preprocess_text(text):
    # 1. Kecilkan huruf & hapus karakter aneh
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 2. Hapus Stopwords (kata sambung)
    text = remover.remove(text)
    # 3. Stemming (Potong imbuhan)
    text = stemmer.stem(text)
    return text

# --- 1. MEMUAT DATABASE (Jurus Pamungkas) ---
@st.cache_data
def load_data():
    # 1. Coba buka file dengan koma, jika gagal pakai titik koma
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip', dtype=str)
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip', dtype=str)
    
    # 2. Jika datanya menyatu di satu kolom, paksa belah dua
    if len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
        
    # 3. PAKSA GANTI NAMA KOLOM (Ini yang akan menghilangkan Error 'uraian')
    # Tidak peduli apa nama aslinya, kolom ke-1 pasti 'kode', ke-2 pasti 'uraian'
    if len(df.columns) >= 2:
        kolom_baru = list(df.columns)
        kolom_baru[0] = 'kode'
        kolom_baru[1] = 'uraian'
        df.columns = kolom_baru
    
    # 4. Bersihkan isi baris dari karakter titik koma (;) di ujung kalimat jika ada terbawa
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")
    
    # 5. Lakukan pembersihan NLP
    df['clean_uraian'] = df['uraian'].apply(preprocess_text)
    
    return df

# --- 2. FITUR HIERARKI ---
def get_hierarchy(kode_target, df):
    parts = str(kode_target).split('.')
    hierarchy_list = []
    current_code = ""
    levels = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]

    for i, part in enumerate(parts):
        current_code = (current_code + "." + part) if current_code else part
        match = df[df['kode'] == current_code]
        uraian = match.iloc[0]['uraian'].title() if not match.empty else "Detail Klasifikasi"
        label = levels[i] if i < len(levels) else f"Level {i+1}"
        hierarchy_list.append(f"└─ **{current_code}**: {uraian} *({label})*")
    return hierarchy_list

# --- 3. LOGIKA NLP & FUZZY MATCHING ---
def smart_classify(user_input, df, top_n=3):
    # A. Preprocess input user
    clean_input = preprocess_text(user_input)
    
    # B. TF-IDF Matching (Akurasi Makna)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    # C. Gabungkan dengan Fuzzy Matching (Toleransi Typo)
    final_scores = []
    for idx, score in enumerate(cosine_sim):
        # Fuzzy ratio antara input user dan uraian asli
        fuzzy_score = fuzz.token_set_ratio(user_input, df.iloc[idx]['uraian']) / 100
        # Gabungkan skor (Bobot: 70% AI, 30% Fuzzy)
        combined_score = (score * 0.7) + (fuzzy_score * 0.3)
        
        if combined_score > 0.1:
            final_scores.append((idx, combined_score))
    
    # Urutkan berdasarkan skor tertinggi
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n]
    
    return final_scores

# --- 4. ANTARMUKA ---
st.title("🗂️ SIKAP (Ultimate)")
st.caption("Asisten Arsiparis Cerdas - Pendekatan NLP & Fuzzy Logic")

try:
    df = load_data()
    user_input = st.text_input("Masukkan Perihal Surat:", placeholder="Contoh: keneikan pangkt pns...")

    if user_input:
        with st.spinner('Menganalisis bahasa...'):
            results = smart_classify(user_input, df)
            
            if results:
                st.write("---")
                for i, (idx, score) in enumerate(results):
                    res = df.iloc[idx]
                    with st.expander(f"Rekomendasi #{i+1}: Kode {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                        st.write("**Struktur Hierarki:**")
                        hierarki = get_hierarchy(res['kode'], df)
                        for h in hierarki:
                            st.markdown(h)
            else:
                st.warning("Tidak ditemukan klasifikasi yang cocok.")
except Exception as e:
    st.error(f"Error: {e}")
