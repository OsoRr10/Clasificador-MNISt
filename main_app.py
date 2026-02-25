import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Configuración de la página
st.set_page_config(page_title="MNIST Classifier Lab", layout="wide")

st.title("🔢 MNIST Digit Classification Lab")
st.markdown("""
Esta aplicación permite explorar cómo diferentes modelos de Machine Learning clasifican dígitos escritos a mano. 
Ajusta los hiperparámetros en la barra lateral y observa los resultados en tiempo real.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data(sample_size=5000):
    # Cargamos una muestra para mantener la app fluida
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data[:sample_size], mnist.target[:sample_size]
    X = X / 255.0  # Normalización
    return X, y

# Barra lateral para configuración global
st.sidebar.header("Configuración de Datos")
sample_size = st.sidebar.slider("Tamaño de la muestra", 1000, 10000, 5000, step=1000)

with st.spinner('Cargando dataset MNIST...'):
    X, y = load_data(sample_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SELECCIÓN DE MODELO ---
st.sidebar.header("Configuración del Modelo")
model_type = st.sidebar.selectbox(
    "Selecciona el Algoritmo",
    ("Logistic Regression", "SVM", "Neural Network (MLP)")
)

params = {}
if model_type == "Logistic Regression":
    params['C'] = st.sidebar.slider("Regularización (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=params['C'], max_iter=100)

elif model_type == "SVM":
    params['C'] = st.sidebar.slider("Regularización (C)", 0.01, 10.0, 1.0)
    params['kernel'] = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf"))
    model = SVC(C=params['C'], kernel=params['kernel'], probability=True)

elif model_type == "Neural Network (MLP)":
    params['hidden_layers'] = st.sidebar.slider("Neuronas en capa oculta", 10, 100, 50)
    params['lr'] = st.sidebar.select_slider("Tasa de aprendizaje", options=[0.001, 0.01, 0.1])
    model = MLPClassifier(hidden_layer_sizes=(params['hidden_layers'],), learning_rate_init=params['lr'], max_iter=200)

# --- ENTRENAMIENTO ---
if st.sidebar.button("🚀 Entrenar y Evaluar"):
    with st.spinner(f'Entrenando {model_type}...'):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # --- MÉTRICAS ---
    st.header(f"📊 Resultados: {model_type}")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred, average='macro'):.2%}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred, average='macro'):.2%}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='macro'):.2%}")

    # --- VISUALIZACIONES ---
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Curvas ROC", "Predicciones Visuales"])

    with tab1:
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Blues")
        st.pyplot(fig_cm)

    with tab2:
        if y_score is not None:
            # Binarizar etiquetas para ROC multiclase
            y_test_bin = label_binarize(y_test, classes=[str(i) for i in range(10)])
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            for i in range(10):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'Dígito {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Falsos Positivos')
            plt.ylabel('Verdaderos Positivos')
            plt.legend(loc="lower right", fontsize='small')
            st.pyplot(fig_roc)
        else:
            st.warning("Este modelo no soporta probabilidades para curvas ROC.")

    with tab3:
        st.subheader("Muestra de Predicciones")
        indices = np.random.choice(len(X_test), 10, replace=False)
        fig_preds, axes = plt.subplots(2, 5, figsize=(12, 6))
        for i, idx in enumerate(indices):
            ax = axes[i//5, i%5]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            color = "green" if y_pred[idx] == y_test[idx] else "red"
            ax.set_title(f"Pred: {y_pred[idx]}\nReal: {y_test[idx]}", color=color)
            ax.axis('off')
        st.pyplot(fig_preds)

    # Gráfico de entrenamiento (Pérdida para MLP)
    if model_type == "Neural Network (MLP)":
        st.subheader("📈 Curva de Pérdida (Loss Curve)")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(model.loss_curve_)
        ax_loss.set_xlabel("Iteraciones")
        ax_loss.set_ylabel("Loss")
        st.pyplot(fig_loss)

else:
    st.info("Configura los parámetros y presiona 'Entrenar y Evaluar' en la barra lateral.")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.caption("Desarrollado con Streamlit y Scikit-learn")
