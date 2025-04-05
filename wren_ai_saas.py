import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import time
import requests

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

# Titre et Introduction
st.title("Expert en Données : Parle à tes Données !")
st.write("""
Cette plateforme vous aide à comprendre vos données et à résoudre des problèmes.
- **Téléchargez vos données** (CSV ou Excel)
- **Personnalisez vos graphiques**
- **Interrogez vos données naturellement**
- **Demandez de l'aide à Datacostexpert le spécialiste de la réduction de coûts*
""")

# Téléchargement et Aperçu des Données
st.header("Télécharge tes Données et Regarde-les")
uploaded_file = st.file_uploader("Choisis ton fichier (CSV ou Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    max_rows = 1000  # Limite le nombre de lignes
    max_file_size_mb = 5  # Limite la taille du fichier en Mo
    max_file_size_bytes = max_file_size_mb * 1024 * 1024  # Convertit les Mo en octets

    if uploaded_file.size > max_file_size_bytes:
        st.error(f"Oups ! Le fichier est trop gros (plus de {max_file_size_mb} Mo). Essaie avec un fichier plus petit.")
    else:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file)

            if len(data) > max_rows:
                st.warning(f"Attention ! Ton fichier a beaucoup de lignes ({len(data)}). On va regarder les {max_rows} premières.")
                data = data.head(max_rows)  # Ne garde que les premières lignes

            # Nettoyage des Données
            data = data.dropna()
            data = data.drop_duplicates()
            data.columns = [col.replace(':', '\\:').strip() for col in data.columns]
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        data[col] = pd.to_numeric(data[col])
                    except ValueError:
                        pass
            if "Unnamed: 0" in data.columns:
                data = data.drop(columns=["Unnamed: 0"])

            # Aperçu des Données
            st.subheader("Voici tes Données :")
            st.dataframe(data.head())

            # Types de Données
            st.subheader("Types de Données :")
            st.write(data.dtypes)

            # Statistiques Descriptives
            st.subheader("Quelques Chiffres :")
            st.write(data.describe())

            # Visualisation des Données
            st.header("Graphiques :")
            numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()

            if numeric_columns:
                selected_column = st.selectbox("Choisis une colonne pour le graphique :", numeric_columns)
                hist_chart = alt.Chart(data).mark_bar().encode(
                    alt.X(selected_column, bin=alt.Bin(maxbins=30)),
                    y='count()'
                ).properties(width=600, height=400).interactive()
                st.altair_chart(hist_chart, use_container_width=True)

                st.subheader("Relations entre les Chiffres :")
                numeric_data = data.select_dtypes(include=['number'])
                if not numeric_data.empty:
                    fig, ax = plt.subplots()
                    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Pas de chiffres à comparer ici.")

            else:
                st.info("Pas de chiffres à montrer en graphique.")

            # Aide de l'IA (Marina AI)
            st.header("Pose tes Questions à Marina AI :")
            user_question = st.text_input("Pose ta question :")

            # Test de la connexion à l'API Mistral AI
            api_url = "https://api.mistral.ai/v1/chat/completions"
            headers = {"Authorization": "Bearer jC7bcWcJBWznK0gFZ8mefGWPgOouBfCK", "Content-Type": "application/json"}

            test_payload = {
                "model": "mistral-tiny",
                "messages": [{"role": "user", "content": "Salut, ça marche ?"}],
                "max_tokens": 10
            }

            print("Test de l'API Mistral AI...")
            try:
                test_response = requests.post(api_url, json=test_payload, headers=headers)
                test_response.raise_for_status()
                print("L'API Mistral AI est OK !")
                print(test_response.json())
            except requests.exceptions.RequestException as e:
                print(f"Erreur avec l'API Mistral AI : {e}")
                print(f"Réponse : {test_response.text if 'test_response' in locals() else 'Rien à dire'}")
            else:
                if st.button("Demande à Marina AI"):
                    if user_question:
                        with st.spinner("Marina AI réfléchit..."):
                            question_directe = f"Calcule et donne la réponse à cette question : {user_question}. Donne uniquement le résultat et un avis concis."

                            payload = {
                                "model": "mistral-tiny",
                                "messages": [{"role": "user", "content": f"Voici les données : {data.head(5).to_csv(index=False)}. {question_directe}"}],
                                "max_tokens": 80,
                                "temperature": 0.3,
                            }
                            retries = 3
                            delay = 5

                            for attempt in range(retries):
                                try:
                                    response = requests.post(api_url, json=payload, headers=headers)
                                    response.raise_for_status()
                                    response_data = response.json()
                                    response_text = response_data.get("choices", [{}])[0].get("message", {}).get("content", "Je ne sais pas quoi dire.")
                                    st.subheader("Marina AI dit :")
                                    st.write(response_text)
                                    break
                                except requests.exceptions.HTTPError as e:
                                    if e.response.status_code == 429 and attempt < retries - 1:
                                        time.sleep(delay)
                                        delay *= 2
                                    else:
                                        st.error(f"Erreur avec l'API Mistral AI : {e}")
                                        if 'response' in locals():
                                            st.error(f"Texte de la réponse : {response.text}")
                                        break
                                except Exception as e:
                                    st.error(f"Une erreur s'est produite : {e}")
                                    break
                            else:
                                st.warning("Pose une question, s'il te plaît.")

        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")

else:
    st.info("Télécharge un fichier CSV ou Excel pour commencer.")
