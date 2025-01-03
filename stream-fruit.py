import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model dan scaler
def load_model_and_scaler(model_file, scaler_file=None):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if scaler_file:
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler

# Fungsi untuk prediksi
def predict_fruit(features, model, scaler=None):
    if scaler:
        features = scaler.transform([features])
    else:
        features = np.array([features])
    prediction_class = model.predict(features)[0]
    return prediction_class

# Konfigurasi Streamlit
st.title("Aplikasi Prediksi Buah")
st.write("Masukkan fitur buah dan pilih model untuk memprediksi jenis buah.")

# Pilih algoritma
algorithm = st.selectbox("Pilih Algoritma", ["Random Forest", "SVM", "Perceptron"])

# Load model dan scaler berdasarkan pilihan algoritma
if algorithm == "Random Forest":
    model_file = 'fruit_RandomForest.pkl'
    model, scaler = load_model_and_scaler(model_file)
elif algorithm == "SVM":
    model_file = 'fruit_SVM.pkl'
    scaler_file = 'scaler_svm.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)
elif algorithm == "Perceptron":
    model_file = 'fruit_Perceptron.pkl'
    scaler_file = 'scaler_perceptron.pkl'
    model, scaler = load_model_and_scaler(model_file, scaler_file)

# Input fitur buah
st.write("Masukkan nilai fitur berikut:")
diameter = st.number_input("Diameter (mm):", value=0.0, step=0.1)
weight = st.number_input("Berat (gram):", value=0.0, step=0.1)
red = st.number_input("Intensitas Merah (0-255):", value=0, step=1)
green = st.number_input("Intensitas Hijau (0-255):", value=0, step=1)
blue = st.number_input("Intensitas Biru (0-255):", value=0, step=1)

input_features = [diameter, weight, red, green, blue]

# Prediksi
if st.button("Prediksi"):
    prediction = predict_fruit(input_features, model, scaler)
    st.success(f"Model memprediksi jenis buah: {prediction}")
