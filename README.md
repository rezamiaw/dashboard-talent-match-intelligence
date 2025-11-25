📘 Deskripsi Proyek

Dashboard ini dikembangkan sebagai bagian dari Studi Kasus Data Analyst 2025 untuk mengidentifikasi pola keberhasilan (Success Pattern) karyawan berperforma tinggi dan mendukung proses pengambilan keputusan HR berbasis data.

Proyek ini terdiri dari tiga tahap utama:
1. Success Pattern Discovery – Analisis kompetensi, kognitif, dan strengths.
2. Talent Match Analysis – Menghitung Final Match Rate per karyawan berdasarkan TGV (Talent Group Variables).
3. Role-based AI Insights – Menggunakan model Grok 4.1 Fast (OpenRouter API) untuk menghasilkan deskripsi peran dan variabel kinerja utama.

⚙️ Tools
Frontend: Streamlit Dashboard
Backend: Supabase (PostgreSQL)
LLM Integration: OpenRouter API – Grok 4.1

Cara Menjalankan Aplikasi

1.Clone repository
git clone https://github.com/rezamiaw/dashboard-talent-match-intelligence.git
cd dashboard-talent-match-intelligence

2.Install dependensi
pip install -r requirements.txt

3.Jalankan Streamlit app
streamlit run app.py

Akses di browser: http://localhost:8501
