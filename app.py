import streamlit as st
import pandas as pd
import re
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from thefuzz import process, fuzz

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="SIKAP - Klasifikasi Arsip Pintar", page_icon="🗂️", layout="wide")

# --- UI & CSS CUSTOM (Warna Elegan Biru & Hijau) ---
st.markdown("""
    <style>
    /* Latar belakang aplikasi dengan gradasi hijau-biru yang sangat lembut */
    .stApp {
        background: linear-gradient(135deg, #f1f8f6 0%, #e1f5fe 100%);
    }
    /* Konfigurasi Judul Utama */
    .sikap-title {
        font-size: 4.5rem; 
        font-weight: 900; 
        color: #004D40; /* Hijau gelap elegan */
        text-align: center;
        margin-bottom: -15px; 
        letter-spacing: 3px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sikap-subtitle {
        font-size: 1.4rem; 
        color: #0277BD; /* Biru pemerintahan */
        text-align: center; 
        font-weight: 600;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }
    /* Desain Expander (Kotak Rekomendasi) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 8px;
        font-weight: bold;
        color: #004D40;
    }
    </style>
""", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE UNTUK FITUR BARU ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# --- INISIALISASI NLP (Sastrawi) ---
@st.cache_resource
def init_nlp():
    stemmer = StemmerFactory().create_stemmer()
    remover = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, remover

stemmer, remover = init_nlp()

# --- KAMUS JARGON & SINGKATAN BIROKRASI ---
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
    "skp": "sasaran kinerja pegawai", 
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

# --- FUNGSI PENERJEMAH SINGKATAN ---
def terjemahkan_singkatan(text):
    kata_kata = str(text).lower().split()
    kata_terjemahan = [kamus_birokrasi.get(kata, kata) for kata in kata_kata]
    return " ".join(kata_terjemahan)

# --- FUNGSI PEMBERSIH UTAMA (Digabung agar tidak tumpang tindih) ---
def preprocess_text(text):
    text = str(text).lower()
    text = terjemahkan_singkatan(text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = remover.remove(text)
    text = stemmer.stem(text)
    return text

# --- 1. MEMUAT DATABASE ---
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
        kolom_baru = list(df.columns)
        kolom_baru[0] = 'kode'
        kolom_baru[1] = 'uraian'
        df.columns = kolom_baru
    
    df['uraian'] = df['uraian'].astype(str).str.replace(r';$', '', regex=True).str.strip().fillna("")
    df['kode'] = df['kode'].astype(str).str.strip().fillna("000")
    df['clean_uraian'] = df['uraian'].apply(preprocess_text)
    return df

# --- 2. FITUR HIERARKI (DENGAN WARNA BERBEDA TIAP LEVEL) ---
def get_hierarchy(kode_target, df):
    parts = str(kode_target).split('.')
    hierarchy_list = []
    current_code = ""
    levels = ["Primer", "Sekunder", "Tersier", "Kuartier", "Kuintier"]
    
    # Palet warna berdasarkan level hierarki (Elegan)
    warna_level = ["#B71C1C", "#1565C0", "#2E7D32", "#E65100", "#4A148C"]

    for i, part in enumerate(parts):
        current_code = (current_code + "." + part) if current_code else part
        match = df[df['kode'] == current_code]
        uraian = match.iloc[0]['uraian'].title() if not match.empty else "Detail Klasifikasi"
        label = levels[i] if i < len(levels) else f"Level {i+1}"
        warna = warna_level[i] if i < len(warna_level) else "#424242"
        
        # Menggunakan HTML untuk mewarnai teks
        html_string = f"<div style='margin-left: {i*20}px; padding: 4px 0;'><span style='color: {warna}; font-weight: bold;'>└─ {current_code}</span> : {uraian} <i style='color: #757575; font-size: 0.9em;'>({label})</i></div>"
        hierarchy_list.append(html_string)
    return hierarchy_list

# --- 3. LOGIKA NLP & FUZZY MATCHING (TIDAK DIUBAH) ---
def smart_classify(user_input, df, top_n=3):
    clean_input = preprocess_text(user_input)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_docs = df['clean_uraian'].tolist() + [clean_input]
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
    
    skor_awal = []
    for idx, score in enumerate(cosine_sim):
        fuzzy_score = fuzz.token_set_ratio(clean_input, df.iloc[idx]['clean_uraian']) / 100
        combined_score = (score * 0.70) + (fuzzy_score * 0.30)
        skor_awal.append({'idx': idx, 'skor': combined_score})
        
    skor_awal = sorted(skor_awal, key=lambda x: x['skor'], reverse=True)
    
    if skor_awal:
        idx_juara_1 = skor_awal[0]['idx']
        kode_juara_1 = str(df.iloc[idx_juara_1]['kode'])
        bagian_kode = kode_juara_1.split('.')
        rumpun_utama = ".".join(bagian_kode[:2]) if len(bagian_kode) >= 2 else bagian_kode[0]
        
        hasil_akhir = []
        for item in skor_awal:
            idx = item['idx']
            skor = item['skor']
            kode_item = str(df.iloc[idx]['kode'])
            if kode_item.startswith(rumpun_utama) and idx != idx_juara_1:
                skor = skor * 1.5 
            hasil_akhir.append((idx, skor))
            
        hasil_akhir = sorted(hasil_akhir, key=lambda x: x[1], reverse=True)
        return hasil_akhir[:top_n]
    return []

# --- 4. ANTARMUKA UTAMA ---

# Judul Aplikasi
st.markdown("<div class='sikap-title'>SIKAP</div>", unsafe_allow_html=True)
st.markdown("<div class='sikap-subtitle'>SIstem Informasi Klasifikasi Arsip Pintar</div>", unsafe_allow_html=True)

try:
    df = load_data()
    
    # Menambahkan Sidebar untuk Riwayat Pencarian
    with st.sidebar:
        st.header("🕒 Riwayat Pencarian")
        if st.session_state.search_history:
            for riwayat in reversed(st.session_state.search_history[-10:]): # Tampilkan 10 terakhir
                st.caption(f"• {riwayat}")
            if st.button("Hapus Riwayat", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.info("Belum ada pencarian.")

    # Membagi Layout menjadi dua Tab
    tab_ai, tab_katalog = st.tabs(["🤖 Pencarian AI (Cerdas)", "📂 Jelajah Kode Klasifikasi (Manual)"])

    # ================= TAB 1: PENCARIAN AI =================
    with tab_ai:
        st.write("Masukkan perihal atau deskripsi surat, biarkan kecerdasan buatan mencari kode klasifikasi yang paling tepat untuk Anda.")
        user_input = st.text_input("📝 Perihal Surat / Dokumen:", placeholder="Contoh: permohonan cuti tahunan pegawai atau undangan rapat tapd...", key="input_ai")

        if user_input:
            # Simpan ke riwayat
            if user_input not in st.session_state.search_history:
                st.session_state.search_history.append(user_input)

            with st.spinner('Menganalisis bahasa dan mencari kecocokan...'):
                results = smart_classify(user_input, df)
                
                if results:
                    st.success("Analisis selesai! Berikut adalah rekomendasi kode untuk dokumen Anda:")
                    
                    pilihan_feedback = [] # Untuk form feedback
                    
                    for i, (idx, score) in enumerate(results):
                        res = df.iloc[idx]
                        pilihan_feedback.append(f"{res['kode']} - {res['uraian'].title()}")
                        
                        # Tampilan Expander yang lebih rapi
                        with st.expander(f"🏅 Rekomendasi #{i+1}: Kode {res['kode']} (Keyakinan: {score:.1%})", expanded=(i==0)):
                            st.markdown(f"**Uraian:** {res['uraian'].title()}")
                            st.markdown("**Struktur Hierarki:**")
                            hierarki = get_hierarchy(res['kode'], df)
                            for h in hierarki:
                                st.markdown(h, unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # --- FITUR FEEDBACK (PEMBELAJARAN) ---
                    st.markdown("#### 💡 Bantu SIKAP Menjadi Lebih Pintar")
                    st.write("Dari rekomendasi di atas, mana kode yang paling tepat menurut Anda?")
                    
                    with st.form("feedback_form"):
                        jawaban_benar = st.radio("Pilih kode yang benar:", pilihan_feedback)
                        submit_feedback = st.form_submit_button("Kirim Masukan")
                        
                        if submit_feedback:
                            # Logika menyimpan feedback (Simulasi penyimpanan data)
                            waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            data_feedback = f"{waktu_sekarang} | Input: {user_input} | Terpilih: {jawaban_benar}\n"
                            
                            # Simpan ke file teks lokal (sebagai log untuk dipelajari nanti)
                            with open("feedback_ai_log.txt", "a", encoding="utf-8") as f:
                                f.write(data_feedback)
                                
                            st.success(f"Terima kasih! Pilihan Anda ({jawaban_benar.split(' - ')[0]}) telah disimpan untuk bahan evaluasi AI ke depannya.")
                else:
                    st.warning("Tidak ditemukan klasifikasi yang cocok. Coba gunakan kata kunci lain.")

    # ================= TAB 2: JELAJAH KODE =================
    with tab_katalog:
        st.write("Cari kode klasifikasi secara manual berdasarkan awalannya.")
        prefix_input = st.text_input("🔍 Masukkan Awalan Kode (Misal: 000, 800, 900.1):", placeholder="Ketik kode di sini...")
        
        if prefix_input:
            # Filter DataFrame berdasarkan string kode yang berawalan dari input user
            hasil_filter = df[df['kode'].str.startswith(prefix_input)]
            
            if not hasil_filter.empty:
                st.success(f"Ditemukan {len(hasil_filter)} kode yang berawalan '{prefix_input}':")
                # Tampilkan dalam bentuk tabel yang elegan
                st.dataframe(
                    hasil_filter[['kode', 'uraian']].rename(columns={'kode': 'Kode Klasifikasi', 'uraian': 'Uraian'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.error(f"Tidak ada kode klasifikasi yang berawalan '{prefix_input}'.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat data: {e}")
