import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Lade das trainierte Modell
best_rf = joblib.load("rf_model.pkl")

# Lade den One-Hot-Encoder und den Scaler
encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Lade die Feature-Spaltennamen
feature_columns = joblib.load("feature_columns.pkl")

# Streamlit-UI
st.title("Vorhersage der Te-Zeit mit Random Forest")

# Eingabefelder für den Nutzer
saegelaenge = st.number_input("Sägelänge", min_value=0.0, value=115.0)
stablaenge = st.number_input("Stablänge", min_value=0.0, value=3710.0)
kg_m = st.number_input("kg/m", min_value=0.0, value=2.907)
verpacken = st.selectbox("Verpacken nach Sägen", ["J", "N"])
kombi = st.selectbox("Kombi-/Stückeloxal", ["K", "N", "S"])
extern = st.selectbox("Von extern", ["J", "N"])
ma = st.selectbox("MA", ["1", "2"])
saegenart = st.selectbox("Sägenart", ["Auto", "Man"])

# Erstelle einen DataFrame für die Eingabedaten
new_data = pd.DataFrame({
    'Sägelänge': [saegelaenge],
    'Stablänge': [stablaenge],
    'kg/m': [kg_m],
    'Verpacken nach Sägen': [verpacken],
    'Kombi-/Stückeloxal': [kombi],
    'von extern': [extern],
    'MA': [ma],
    'Sägenart': [saegenart]
})

# Wandle kategorische Variablen um (One-Hot-Encoding)
categorical_columns = ['Verpacken nach Sägen', 'Kombi-/Stückeloxal', 'von extern', 'MA', 'Sägenart']
X_categorical_encoded = encoder.transform(new_data[categorical_columns])
X_categorical_df = pd.DataFrame(X_categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns))

# Numerische Features skalieren
X_numerical_scaled = scaler.transform(new_data[['Sägelänge', 'Stablänge', 'kg/m']])
X_numerical_df = pd.DataFrame(X_numerical_scaled, columns=['Sägelänge', 'Stablänge', 'kg/m'])

# Kombiniere numerische und kategorische Merkmale
X_processed = pd.concat([X_numerical_df, X_categorical_df], axis=1)

# Stelle sicher, dass alle Features in der richtigen Reihenfolge sind
X_processed = X_processed.reindex(columns=feature_columns, fill_value=0)

# Vorhersage durchführen
if st.button("Vorhersage starten"):
    te_pred = best_rf.predict(X_processed)
    st.write(f"**Vorhergesagte TE-Zeit:** {te_pred[0]:.4f}")
