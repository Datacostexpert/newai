import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import time
import requests
import os
import numpy as np

# Style CSS personnalisé
custom_css = """
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
h1 {
    color: #4CAF50;
    font-size: 2.5em;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Titre
st.title("Expert en Données : Interrogez vos Données et réduisez vos dépenses !")
st.write("""
Cette plateforme t'aide à comprendre tes données et à résoudre des problèmes.
- **Téléchargez vos données** (CSV ou Excel)
- **Visualisez et nettoyez vos données**
- **Interrogez-les avec l'IA**
""")

# Fichier Upload
st.header("📁 Téléchargez vos Données")
uploaded_file_main = st.file_uploader("Choisissez un fichier (CSV ou Excel)", type=["csv", "xlsx", "xls"])

with st.sidebar:
    st.title("🔧 Paramètres")
    uploaded_file_sidebar = st.file_uploader("Ou téléchargez un fichier ici", type=["csv", "xlsx", "xls"])

uploaded_file = uploaded_file_sidebar if uploaded_file_sidebar else uploaded_file_main
data = None

if uploaded_file:
    try:
        ext = os.path.splitext(uploaded_file.name)[1]
        if ext in [".xlsx", ".xls"]:
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file, on_bad_lines='skip')
        st.toast("Fichier chargé avec succès ✅", icon="📥")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        data = None

if data is not None:
    st.header("📋 Aperçu des Données")
    def highlight_missing(val):
        return 'background-color: red' if pd.isnull(val) else ''
    st.dataframe(data.head().style.applymap(highlight_missing))

    # Nettoyage
    initial_rows = len(data)
    data = data.dropna()
    rows_dropped_na = initial_rows - len(data)
    initial_rows = len(data)
    data = data.drop_duplicates()
    rows_dropped_duplicates = initial_rows - len(data)
    data.columns = [col.replace(':', '\\:').strip() for col in data.columns]
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                pass
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # Résumé IA
    if st.checkbox("🧠 Obtenir un résumé automatique des données"):
        with st.spinner("Génération du résumé..."):
            sample_json = data.head(10).to_json(orient='records')
            prompt = f"Voici un échantillon des données : {sample_json}. Fais un résumé rapide : types, tendances, anomalies."
            headers = {"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}", "Content-Type": "application/json"}
            payload = {
                "model": "mistral-tiny",
                "messages": [
                    {"role": "system", "content": "Vous êtes un analyste de données expérimenté."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300
            }
            response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                summary = response.json()["choices"][0]["message"]["content"]
                st.markdown("### 📾 Résumé IA")
                st.write(summary)
            else:
                st.error("Erreur de réponse de Mistral AI.")

    st.subheader("🔍 Types de Données")
    st.write(data.dtypes)

    st.subheader("❓ Valeurs Manquantes")
    st.write(data.isnull().sum())

    st.subheader("📊 Statistiques Descriptives")
    st.write(data.describe())

    st.header("📈 Visualisation des Données")
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

    if numeric_columns:
        selected_column = st.selectbox("📌 Colonne pour histogramme", numeric_columns)
        hist_chart = alt.Chart(data).mark_bar().encode(
            alt.X(selected_column, bin=alt.Bin(maxbins=30)),
            y='count()'
        ).interactive()
        st.altair_chart(hist_chart, use_container_width=True)

        if len(numeric_columns) >= 2:
            x_col = st.selectbox("X", numeric_columns, key="x")
            y_col = st.selectbox("Y", numeric_columns, key="y")
            scatter_chart = alt.Chart(data).mark_circle(size=60).encode(
                x=x_col, y=y_col, tooltip=[x_col, y_col]
            ).interactive()
            st.altair_chart(scatter_chart, use_container_width=True)

        st.subheader("🔗 Corrélation entre variables")
        fig, ax = plt.subplots()
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Aucune colonne numérique détectée.")

    if categorical_columns:
        selected_cat = st.selectbox("Colonne catégorielle", categorical_columns)
        st.bar_chart(data[selected_cat].value_counts())

    st.subheader("💾 Télécharger les Données Nettoyées")
    cleaning_summary = ""
    if rows_dropped_na > 0:
        cleaning_summary += f"- {rows_dropped_na} lignes avec valeurs manquantes supprimées\n"
    if rows_dropped_duplicates > 0:
        cleaning_summary += f"- {rows_dropped_duplicates} doublons supprimés\n"
    cleaning_summary += "- Tentative de conversion des textes en chiffres\n"
    st.info(cleaning_summary)
    st.download_button("📥 Télécharger les données nettoyées", data.to_csv(index=False), "donnees_nettoyees.csv")

    # Assistance IA
    st.header("🤖 Posez une question aux données")
    user_question = st.text_input("Votre question :")
    if st.button("Demander à Mistral AI"):
        if user_question:
            with st.spinner("Analyse en cours..."):
                headers = {"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}", "Content-Type": "application/json"}
                data_json = data.head(100).to_json(orient='records')
                payload = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system", "content": "Vous êtes un analyste de données."},
                        {"role": "user", "content": f"Données : {data_json}. Question : {user_question}"}
                    ],
                    "max_tokens": 300
                }
                response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    st.subheader("💬 Réponse IA")
                    st.write(answer)
                else:
                    st.error("Erreur avec l’API Mistral.")
        else:
            st.warning("Entrez une question pour continuer.")
else:
    st.info("Veuillez charger un fichier CSV ou Excel pour commencer.")
