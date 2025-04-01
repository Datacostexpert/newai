import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# CSS personnalisé
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

# Titre et Introduction
st.title("Data Cost Expert : Discutez avec vos données")
st.write("""
Cette plateforme permet d'analyser facilement les données et de résoudre des problèmes métier.
- **Téléchargez vos données** (format CSV)
- **Visualisez des insights**
- **Posez des questions** en langage naturel
- **Contactez un expert en cas de besoin**
""")

# Téléchargement & Aperçu des Données
st.header("Téléchargement & Aperçu des Données")
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, on_bad_lines='skip')

    # Clean up column names
    data.columns = [col.replace(':', '\\:') if ':' in col else col for col in data.columns]

    # Clean up data types
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col])
            except ValueError:
                pass

    # Remove the "Unnamed: 0" column if it exists
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    # Aperçu des données
    st.subheader("Aperçu des Données")
    st.dataframe(data.head())
    
    # Types de données
    st.subheader("Types de Données")
    st.write(data.dtypes)
    
    # Valeurs manquantes
    st.subheader("Valeurs Manquantes")
    missing_values = data.isnull().sum()
    st.write(missing_values[missing_values > 0])
    
    # Statistiques Descriptives
    st.subheader("Statistiques Descriptives")
    st.write(data.describe())
    
    # Visualisation des Données
    st.header("Visualisation des Données")
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    
    if numeric_columns:
        selected_column = st.selectbox("Sélectionnez une colonne pour l'histogramme", numeric_columns)
        hist_chart = alt.Chart(data).mark_bar().encode(
            alt.X(selected_column, bin=alt.Bin(maxbins=30)),
            y='count()'
        ).properties(width=600, height=400).interactive()
        st.altair_chart(hist_chart, use_container_width=True)
        
        st.subheader("Corrélation entre Variables")
        # Sélectionne uniquement les colonnes numériques pour la corrélation
        numeric_data = data.select_dtypes(include=['number'])
        if not numeric_data.empty: # check if numeric_data is empty
            fig, ax = plt.subplots()
            sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Aucune colonne numérique disponible pour la corrélation.")
        
    else:
        st.info("Aucune colonne numérique disponible pour la visualisation.")
    
    # Assistance IA avec API Mistral AI (gratuite)
    st.header("Assistance IA")
    user_question = st.text_input("Posez une question sur les données :")
    if st.button("Demander à Marina AI"):
        if user_question:
            with st.spinner("Analyse en cours..."):
                api_url = "https://api.mistral.ai/v1/chat/completions"
                headers = {"Authorization": "Bearer jC7bcWcJBWznK0gFZ8mefGWPgOouBfCK", "Content-Type": "application/json"}
                data_string = data.to_csv(index=False) # Convertit le dataframe en string (CSV)
                payload = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": f"Voici les données : {data_string}. {user_question}"}],
                    "max_tokens": 100
                }
                response = requests.post(api_url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Pas de réponse disponible.")
                    st.subheader("Réponse de Marina AI")
                    st.write(response_text)
                else:
                    st.error("Erreur lors de la communication avec l'API Mistral AI.")
        else:
            st.warning("Veuillez entrer une question.")
else:
    st.info("Veuillez télécharger un fichier CSV pour commencer l'analyse.")