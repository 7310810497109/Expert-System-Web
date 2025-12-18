import pickle
import numpy as np
import os

# Lokasi model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/dental_model.pkl')

# Kita butuh kamus data ini agar tetap konsisten
GEJALA_DICT = {
    'G01': 'Gigi ngilu', 'G02': 'Gigi berdenyut', 'G03': 'Gigi goyang',
    'G04': 'Gigi baru muncul, gigi lama masih ada', 'G05': 'Gusi bengkak',
    'G06': 'Gigi berlubang', 'G07': 'Pipi bengkak dan terasa hangat',
    'G08': 'Nyeri saat mengunyah', 'G09': 'Sariawan', 'G10': 'Sakit gigi bungsu',
    'G11': 'Gigi berlubang tanpa sakit', 'G12': 'Gigi sakit saat diketuk',
    'G13': 'Radang', 'G14': 'Karang gigi'
}

PENYAKIT_DICT = {
    'P01': 'Pulpitis Irreversible', 'P02': 'Pulpitis Reversible',
    'P03': 'Periodontitis', 'P04': 'Cellulitis and abscess of mouth',
    'P05': 'Periapical abscess without sinus', 'P06': 'Carries limmited to enamel',
    'P07': 'Persis Tensi', 'P08': 'Stomatitis', 'P09': 'Impaksi',
    'P10': 'Acute apical periodontitis of pulpa origin', 'P11': 'Necrosis of pulp',
    'P12': 'Gingitivis kronis'
}

FITUR_GEJALA = list(GEJALA_DICT.keys())

def load_model():
    """Memuat model yang sudah dilatih"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        return None

def prediksi_penyakit(list_gejala_input):
    """
    Menerima list kode gejala ['G01', 'G03']
    Mengembalikan kode penyakit, nama penyakit, dan probabilitas
    """
    model = load_model()
    if not model:
        return "Error", "Model tidak ditemukan", []

    # Konversi input user ke vektor biner (0/1)
    input_vector = [0] * len(FITUR_GEJALA)
    for g in list_gejala_input:
        if g in FITUR_GEJALA:
            idx = FITUR_GEJALA.index(g)
            input_vector[idx] = 1
    
    input_array = np.array([input_vector])
    
    # Prediksi
    prediksi_kode = model.predict(input_array)[0]
    probs = model.predict_proba(input_array)[0]
    
    nama_penyakit = PENYAKIT_DICT.get(prediksi_kode, "Tidak Diketahui")
    
    # Mengambil urutan kelas penyakit dari model untuk visualisasi nanti
    classes = model.classes_
    hasil_probs = []
    for i, cls in enumerate(classes):
        hasil_probs.append({
            'kode': cls,
            'nama': PENYAKIT_DICT.get(cls, cls),
            'probabilitas': round(probs[i], 4)
        })
        
    return prediksi_kode, nama_penyakit, hasil_probs