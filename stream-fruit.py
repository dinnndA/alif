import streamlit as st
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

file = 'fruit.xlsx'
fruit = pd.read_excel(file)
x = fruit[['diameter', 'weight', 'red', 'green', 'blue']]
y = fruit['name']

label_to_class = {'grapefruit': 0, 'orange': 1}
class_to_label = {v: k for k, v in label_to_class.items()}

def load_model_and_scaler(model_file, scaler_file=None):
    with open('scaler_perceptron.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = None
    if scaler_file:
        with open('scaler_svm.pkl', 'rb') as f:
            scaler = pickle.load(f)
    return model, scaler

def predict_fruit(features, model, scaler=None):
    if scaler:
        features = scaler.transform([features])
    prediction_class = model.predict([features])[0]  # Prediksi kelas
    prediction_label = class_to_label[prediction_class]  # Mapping ke label
    return prediction_label, prediction_class

st.title("Aplikasi Prediksi Buah Dengan Model")
st.write("Pilih algoritma yang digunakan, masukkan fitur buah untuk memprediksi jenis buah.")

algorithm = st.selectbox("Pilih Algoritma", ["Random Forest", "SVM", "Perceptron"])

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

input_features = []
for col in x.columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
    input_features.append(value)

if st.button("Prediksi"):
    if not all(input_features):  # Validasi input
        st.error("Harap masukkan semua nilai fitur.")
    else:
        label, class_index = predict_fruit(input_features, model, scaler)
        st.success(f"Model memprediksi jenis buah: {label} (Cluster: {class_index})")

st.write("Fitur yang dimasukkan:", input_features)
st.write("Bentuk input sebelum prediksi:", len(input_features))