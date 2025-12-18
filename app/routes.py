from flask import render_template, request, url_for
from app import app
from app.model_logic import prediksi_penyakit, GEJALA_DICT
import matplotlib
matplotlib.use('Agg') # Penting: Mode non-GUI agar tidak error di server web
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

@app.route('/')
def index():
    # Kirim data gejala ke HTML untuk dibuat checkbox
    return render_template('index.html', gejala_dict=GEJALA_DICT)

@app.route('/diagnosa', methods=['POST'])
def diagnosa():
    # 1. Ambil data dari form (checkbox yang dicentang)
    gejala_terpilih = request.form.getlist('gejala')
    
    if not gejala_terpilih:
        return render_template('index.html', gejala_dict=GEJALA_DICT, error="Pilih setidaknya satu gejala!")

    # 2. Minta model melakukan prediksi
    kode_penyakit, nama_penyakit, data_probabilitas = prediksi_penyakit(gejala_terpilih)

    # 3. Buat Visualisasi Grafik (Chart)
    # Urutkan data agar grafik rapi (tertinggi di atas)
    data_sorted = sorted(data_probabilitas, key=lambda x: x['probabilitas'], reverse=True)
    nama_p = [x['nama'] for x in data_sorted]
    nilai_p = [x['probabilitas'] for x in data_sorted]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=nilai_p, y=nama_p, palette='viridis')
    plt.title('Probabilitas Diagnosa Penyakit')
    plt.xlabel('Nilai Probabilitas (0-1)')
    plt.xlim(0, 1.0)
    plt.tight_layout()

    # Simpan grafik ke dalam memory (bukan file fisik) agar lebih cepat & aman
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() # Tutup plot agar hemat memori

    # 4. Tampilkan halaman hasil
    return render_template('result.html', 
                           penyakit=nama_penyakit, 
                           kode=kode_penyakit, 
                           gejala_input=gejala_terpilih,
                           gejala_dict=GEJALA_DICT,
                           plot_url=plot_url)