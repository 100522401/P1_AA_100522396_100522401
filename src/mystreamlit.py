import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

st.set_page_config(page_title="Predictor de contrataciones de depósitos a plazos", page_icon="💰", layout="centered")
st.title(" Predicción de Suscripción a Depósito a plazos")
st.write(""" Esta aplicación evalúa si un cliente tiene probabilidades de suscribir un depósito a plazo.
          Para ello utiliza el modelo final generado en la práctica 
         """)



def transform_pdays(X):
  X = X.copy()
  X["contacted_before"] = (X["pdays"] != -1).astype(int)
  X["pdays"] = X["pdays"].replace(-1, np.nan)
  return X

#Carga del paquete del modelo
@st.cache_resource
def load_pack():
    return load("notebooks/modelo_final.joblib")

try:
    pack = load_pack()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Extraer componentes del paquete
pipeline = pack["pipeline"]
feature_metadata = pack["feature_metadata"]
classes_ = pack.get("classes_", [])

st.markdown("### Introduce los valores de las variables:")

# Usamos un formulario para evitar recalcular la predicción con cada pulsación
with st.form("prediction_form"):
    inputs = {}
    
    # Renderizamos las variables numéricas
    st.subheader("Variables Numéricas")
    num_cols = st.columns(2)
    num_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "numerical":
            with num_cols[num_idx % 2]:
                med = float(meta.get("median", 0.0))
                # Usamos number_input para entradas numéricas
                inputs[feat] = st.number_input(
                    label=feat,
                    min_value=float(meta.get("min", -1e9)),
                    max_value=float(meta.get("max", 1e9)),
                    value=med,
                    step=1.0 if med.is_integer() else 0.1
                )
            num_idx += 1

    # Renderizamos las variables categóricas
    st.subheader("Variables Categóricas")
    cat_cols = st.columns(2)
    cat_idx = 0
    
    for feat, meta in feature_metadata.items():
        if meta["type"] == "categorical":
            with cat_cols[cat_idx % 2]:
                opts = meta.get("options", [])
                inputs[feat] = st.selectbox(
                    label=feat,
                    options=opts,
                    index=0 if opts else None
                )
            cat_idx += 1
            
    st.markdown("---")
    submitted = st.form_submit_button("Predecir", use_container_width=True)

if submitted:
    # Convertimos las entradas en un DataFrame de una única fila
    X_new = pd.DataFrame([inputs])
    try:
        # Predecimos usando el Pipeline (que ya incorpora todo el preprocesamiento)
        y_pred = pipeline.predict(X_new)[0]
        
        if y_pred == 1:
            st.success(f"### Predicción: YES")
        else:
            st.error(f"### Predicción: NO")
            
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")