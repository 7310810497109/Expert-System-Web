import pickle
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import os

# --- DEFINISI DATA ---
gejala_dict = {
    'G01': 'Gigi ngilu', 'G02': 'Gigi berdenyut', 'G03': 'Gigi goyang',
    'G04': 'Gigi baru muncul, gigi lama masih ada', 'G05': 'Gusi bengkak',
    'G06': 'Gigi berlubang', 'G07': 'Pipi bengkak dan terasa hangat',
    'G08': 'Nyeri saat mengunyah', 'G09': 'Sariawan', 'G10': 'Sakit gigi bungsu',
    'G11': 'Gigi berlubang tanpa sakit', 'G12': 'Gigi sakit saat diketuk',
    'G13': 'Radang', 'G14': 'Karang gigi'
}

knowledge_base = {
    'P01': ['G02', 'G06'], 'P02': ['G01', 'G06'], 'P03': ['G03'],
    'P04': ['G07', 'G13', 'G06'], 'P05': ['G05', 'G06'], 'P06': ['G06', 'G11'],
    'P07': ['G04'], 'P08': ['G09'], 'P09': ['G10'],
    'P10': ['G02', 'G08', 'G12','G06'], 'P11': ['G06', 'G11'], 'P12': ['G14']
}

# --- PROSES TRAINING ---
X_train = []
y_train = []
fitur_gejala = list(gejala_dict.keys())

print("Sedang melatih model...")

for penyakit, gejala_terkait in knowledge_base.items():
    row = [0] * len(fitur_gejala)
    for g in gejala_terkait:
        if g in fitur_gejala:
            index = fitur_gejala.index(g)
            row[index] = 1
    
    # Duplikasi data 5x agar model stabil
    for _ in range(5):
        X_train.append(row)
        y_train.append(penyakit)

X_train = np.array(X_train)
y_train = np.array(y_train)

model = MultinomialNB()
model.fit(X_train, y_train)

# --- SIMPAN MODEL ---
# Pastikan folder models ada
if not os.path.exists('models'):
    os.makedirs('models')

# Simpan model ke file .pkl
with open('models/dental_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("SUKSES! Model disimpan di: models/dental_model.pkl")