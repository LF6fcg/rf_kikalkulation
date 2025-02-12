import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import requests
from io import BytesIO
import hashlib

# Benutzername und Passwort (sicherer wäre eine verschlüsselte Speicherung)
USERNAME = 'PRA'
PASSWORD = 'ki_gaex'

# Funktion zum Hashen des Passworts
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Beispiel-Hash für das gespeicherte Passwort (stelle sicher, dass du dein Passwort richtig hasthenst)
stored_password_hash = hash_password(PASSWORD)

# Funktion für das Login
def login():
    st.title('Login')
    username_input = st.text_input('Benutzername')
    password_input = st.text_input('Passwort', type='password')

    if st.button('Login'):
        if username_input == USERNAME and hash_password(password_input) == stored_password_hash:
            st.session_state.logged_in = True
            st.success('Erfolgreich eingeloggt!')
        else:
            st.session_state.logged_in = False
            st.error('Falscher Benutzername oder Passwort!')

def load_file_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Überprüfen, ob der Request erfolgreich war (Statuscode 200)
        file_content = BytesIO(response.content)  # Inhalt der Datei in Bytes umwandeln
        return joblib.load(file_content)  # Datei mit joblib laden (vorausgesetzt, sie ist im Pickle-Format)
    except requests.exceptions.RequestException as e:
        st.error(f"Fehler beim Laden der Datei: {url}")
        st.error(str(e))
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden der Datei: {url}")
        st.error(str(e))
        return None
 
#------------------------------------------------------------------------------------------------------------------------

# Funktion für die App, die nur zugänglich ist, wenn der Benutzer eingeloggt ist
def main_app():
    # URLs zu den Dateien auf GitHub (ersetze mit deinen echten Links)
    model_url = "https://raw.githubusercontent.com/LF6fcg/rf_kikalkulation/main/streamlit_app/rf_model.pkl"
    encoder_url = "https://raw.githubusercontent.com/LF6fcg/rf_kikalkulation/main/streamlit_app/onehot_encoder.pkl"
    scaler_url = "https://raw.githubusercontent.com/LF6fcg/rf_kikalkulation/main/streamlit_app/scaler.pkl"
    feature_columns_url = "https://raw.githubusercontent.com/LF6fcg/rf_kikalkulation/main/streamlit_app/feature_columns.pkl"
    
    best_rf = load_file_from_github(model_url)
    encoder = load_file_from_github(encoder_url)
    scaler = load_file_from_github(scaler_url)
    feature_columns = load_file_from_github(feature_columns_url)

    # Debugging: Überprüfe, ob alles geladen wurde
    st.write(f"Model geladen: {best_rf is not None}")
    st.write(f"Encoder geladen: {encoder is not None}")
    st.write(f"Scaler geladen: {scaler is not None}")
    st.write(f"Feature Columns geladen: {feature_columns is not None}")
    
    # Dein Streamlit Code für die App geht hier weiter...
    
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

# Logik für den Login-Status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    
if st.session_state.logged_in:
    main_app()  # Wenn eingeloggt, zeige die App
else:
    login()  # Wenn nicht eingeloggt, zeige das Login-Formular
