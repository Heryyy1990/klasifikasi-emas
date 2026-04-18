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

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-card { background-color: white; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- INISIALISASI NLP ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

# --- 1. KAMUS FRASA (PENCEGAH SALAH ARAH KELAS BERAT) ---
kamus_frasa = {
    # Kepegawaian (800)
    "kenaikan gaji berkala": "penggajian pegawai",
    "gaji berkala": "penggajian pegawai",
    "analisis jabatan": "anjab kepegawaian",
    "analisis beban kerja": "abk kepegawaian",
    "daftar urut kepangkatan": "duk pegawai",
    "ijin belajar": "pendidikan lanjutan pegawai",
    "izin belajar": "pendidikan lanjutan pegawai",
    "tugas belajar": "pendidikan lanjutan pegawai",
    "cuti tahunan": "cuti pegawai",
    "cuti melahirkan": "cuti pegawai",
    "cuti sakit": "cuti pegawai",
    "cuti besar": "cuti pegawai",
    "pakaian dinas": "seragam perlengkapan pegawai",
    "pelantikan pejabat": "jabatan struktural",
    "mutasi masuk": "pindah tugas pegawai",
    "mutasi keluar": "pindah tugas pegawai",
    
    # Keuangan & Perencanaan (900 & 000)
    "rencana kerja anggaran": "rka keuangan",
    "dokumen pelaksanaan anggaran": "dpa keuangan",
    "surat permintaan pembayaran": "spp pencairan",
    "surat perintah membayar": "spm pencairan",
    "surat perintah pencairan dana": "sp2d pencairan",
    "uang persediaan": "up bendahara",
    "ganti uang": "gu bendahara",
    "tambah uang": "tu bendahara",
    "tuntutan ganti rugi": "tgr keuangan",
    "pajak daerah": "pendapatan retribusi",
    
    # Umum, Perlengkapan & Perjalanan (000 & 010)
    "perjalanan dinas": "sppd kunjungan kerja",
    "alat tulis kantor": "atk perlengkapan kantor",
    "kendaraan dinas": "mobil dinas kendaraan",
    "aset daerah": "barang milik daerah",
    "serah terima": "penyerahan bast",
    "rapat koordinasi": "rakor rapat",
    "rapat pimpinan": "rapim pimpinan",
    "standar operasional prosedur": "sop ketatalaksanaan",
    
    # Hukum, Pengawasan & Pemerintahan Daerah (100 & 700)
    "tindak pidana korupsi": "tipikor pidana",
    "hak asasi manusia": "ham hukum",
    "bantuan hukum": "advokasi hukum",
    "rapat paripurna": "paripurna dprd",
    "tanda daftar": "perizinan pendaftaran",
    "izin mendirikan bangunan": "imb perizinan",
    
    # Kesejahteraan, Sosial & Pembangunan
    "bantuan sosial": "hibah bansos",
    "air minum": "spam perumahan",
    "tenaga kerja": "ketenagakerjaan",
    "hari libur": "hari libur nasional"
}
}

# --- 2. KAMUS JARGON BIROKRASI (VERSI FULL PERMENDAGRI 83/2022) ---
kamus_birokrasi = {
    "apbd": "anggaran pendapatan dan belanja daerah",
    "apbn": "anggaran pendapatan dan belanja negara",
    "tapd": "tim anggaran pemerintah daerah",
    "dpa": "dokumen pelaksanaan anggaran",
    "rka": "rencana kerja anggaran",
    "skpd": "satuan kerja perangkat daerah",
    "ppkd": "pejabat pengelola keuangan daerah",
    "ppa": "prioritas plafon anggaran",
    "spp": "surat permintaan pembayaran",
    "spm": "surat perintah membayar",
    "sp2d": "surat perintah pencairan dana",
    "up": "uang persediaan",
    "gu": "ganti uang",
    "tu": "tambah uang",
    "ls": "langsung",
    "bud": "bendahara umum daerah",
    "bku": "buku kas umum",
    "sakd": "sistem akuntansi keuangan daerah",
    "phln": "pinjaman hibah luar negeri",
    "bumd": "badan usaha milik daerah",
    "blud": "badan layanan umum daerah",
    "dau": "dana alokasi umum",
    "dak": "dana alokasi khusus",
    "dbh": "dana bagi hasil",
    "asn": "aparatur sipil negara",
    "pns": "pegawai negeri sipil",
    "cpns": "calon pegawai negeri sipil",
    "pppk": "pegawai pemerintah dengan perjanjian kerja",
    "p3k": "pegawai pemerintah dengan perjanjian kerja",
    "nip": "nomor induk pegawai",
    "bkn": "badan kepegawaian negara",
    "skp": "standar kinerja pegawai",
    "duk": "daftar urut kepangkatan",
    "karpeg": "kartu pegawai",
    "kpe": "kartu pegawai elektronik",
    "karis": "kartu istri",
    "karsu": "kartu suami",
    "lp2p": "laporan pajak penghasilan pribadi",
    "kp4": "keterangan penerimaan pembayaran penghasilan pegawai",
    "baperjakat": "badan pertimbangan jabatan dan pangkat",
    "bpjs": "badan penyelenggara jaminan sosial",
    "diklat": "pendidikan dan pelatihan",
    "bimtek": "bimbingan teknis",
    "lhp": "laporan hasil pemeriksaan",
    "lha": "laporan hasil audit",
    "lhpo": "laporan hasil pemeriksaan operasional",
    "lhe": "laporan hasil evaluasi",
    "lhai": "laporan hasil audit investigasi",
    "tpk": "tindak pidana korupsi",
    "gcg": "good corporate governance",
    "perda": "peraturan daerah",
    "perbup": "peraturan bupati",
    "perwali": "peraturan wali kota",
    "mou": "memorandum of understanding nota kesepakatan",
    "sop": "standar operasional prosedur",
    "haki": "hak atas kekayaan intelektual",
    "dprd": "dewan perwakilan rakyat daerah",
    "musrenbang": "musyawarah perencanaan pembangunan",
    "lkpj": "laporan keterangan pertanggungjawaban",
    "lppd": "laporan penyelenggaraan pemerintahan daerah",
    "bmd": "barang milik daerah",
    "kak": "kerangka acuan kerja",
    "sppd": "surat perintah perjalanan dinas",
    "spt": "surat perintah tugas",
    "nodin": "nota dinas",
    "bap": "berita acara pemeriksaan",
    "bast": "berita acara serah terima",
    "kpu": "komisi pemilihan umum",
    "kpud": "komisi pemilihan umum daerah",
    "dp4": "daftar penduduk potensial pemilih",
    "dps": "daftar pemilih sementara",
    "dpt": "daftar pemilih tetap",
    "panwasda": "panitia pengawas daerah",
    "ppk": "panitia pemilihan kecamatan",
    "pps": "panitia pemungutan suara",
    "kpps": "kelompok penyelenggara pemungutan suara",
    "ormas": "organisasi kemasyarakatan",
    "lsm": "lembaga swadaya masyarakat",
    "parpol": "partai politik",
    "anri": "arsip nasional republik indonesia",
    "jra": "jadwal retensi arsip",
    "sikn": "sistem informasi kearsipan nasional",
    "jikn": "jaringan informasi kearsipan nasional",
    "spam": "sistem penyediaan air minum",
    "psat": "pangan segar asal tumbuhan",
    "bumdes": "badan usaha milik desa",
    "bos": "bantuan operasional sekolah",
    "paud": "pendidikan anak usia dini",
    "rtrw": "rencana tata ruang wilayah",
    "rdtr": "rencana detail tata ruang",
    "rtbl": "rencana tata bangunan dan lingkungan",
    "amdal": "analisis mengenai dampak lingkungan",
    "ukl": "upaya pengelolaan lingkungan",
    "upl": "upaya pemantauan lingkungan",
    "b3": "bahan berbahaya dan beracun",
    "sar": "search and rescue pencarian dan pertolongan"
}

def terjemahkan_teks(text):
    text = str(text).lower()
    for frasa, pengganti in kamus_frasa.items():
        if frasa in text:
            text = text.replace(frasa, pengganti)
    kata_kata = text.split()
    text = " ".join([kamus_birokrasi.get(k, k) for k in kata_kata])
    return text

def preprocess_text(text):
    text = terjemahkan_teks(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = remover.remove(text)
    text = stemmer.stem(text)
    return text

# --- MANAJEMEN DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=',', on_bad_lines='skip', dtype=str)
    except:
        df = pd.read_csv('klasifikasi_arsip_emas.csv', sep=';', on_bad_lines='skip', dtype=str)
    
    if len(df.columns) == 1:
        col_name = df.columns[0]
        df[['kode', 'uraian']] = df[col_name].str.split(r'[,;]', n=1, expand=True)
        df = df.drop(columns=[col_name])
        
    if len(df.columns) >= 2:
        df.columns = ['kode', 'uraian'] + list(df.columns[2:])
    
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")
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

# --- FITUR POHON HIERARKI ---
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

# --- LOGIKA PENCARIAN (BEBAS GEMBOK & SET RATIO) ---
def smart_classify(user_input, df, top_n=3):
    clean_input = preprocess_text(user_input)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    final_scores = []
    for idx, score in enumerate(cosine_sim):
        fuzzy_score = fuzz.token_set_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        combined_score = (score * 0.75) + (fuzzy_score * 0.25)
        final_scores.append((idx, combined_score))
        
    return sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_n]

# --- UI DASHBOARD ---
st.sidebar.title("🏢 SIKAP Muna Barat")
menu = st.sidebar.radio("Navigasi:", ["🔍 Pencarian Pintar", "📖 Kamus Kode", "📊 Laporan Strategis"])

df = load_data()

if menu == "🔍 Pencarian Pintar":
    st.title("🔍 Pencarian Klasifikasi")
    user_input = st.text_input("Ketik Perihal Surat:", placeholder="Misal: Kenaikan gaji berkala...")
    if user_input:
        results = smart_classify(user_input, df)
        for i, (idx, score) in enumerate(results):
            res = df.iloc[idx]
            with st.container():
                st.markdown(f'<div class="result-card"><h4>🏆 #{i+1} | Kode: {res["kode"]} ({score:.1%})</h4><p>{res["uraian"]}</p></div>', unsafe_allow_html=True)
                with st.expander("🌳 Lihat Pohon Hierarki"):
                    for h in get_hierarchy(res['kode'], df):
                        st.markdown(h)
                if st.button(f"✅ Pilih Kode {res['kode']}", key=f"btn_{idx}"):
                    save_feedback(user_input, res['kode'])
                    st.success("Data tersimpan untuk laporan.")

elif menu == "📖 Kamus Kode":
    st.title("📖 Kamus Kode")
    search_code = st.text_input("Masukkan Kode (Misal: 900):")
    if search_code:
        f_df = df[df['kode'].str.startswith(search_code)]
        st.table(f_df[['kode', 'uraian']])

elif menu == "📊 Laporan Strategis":
    st.title("📊 Laporan Tren")
    if os.path.isfile('log_penggunaan.csv'):
        log_df = pd.read_csv('log_penggunaan.csv')
        st.metric("Total Arsip", len(log_df))
        st.bar_chart(log_df['Kode_Dipilih'].value_counts().head(5))
        st.dataframe(log_df.tail(10))
