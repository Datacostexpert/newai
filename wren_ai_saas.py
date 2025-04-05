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
st.title("Expert en Données : Interrogez vos Données et optimiser votre entreprise !")
st.write("""
Cette plateforme t'aide à comprendre tes données et à résoudre des problèmes.
- **Télécharge tes données** (CSV ou Excel)
- **Personnalise tes graphiques**
- **Interroge tes données naturellement**
- *Demande de l'aide à Datacostexpert le spécialiste de la réduction de coûts*
""")

# Téléchargement et Aperçu des Données
st.header("Téléchargez vos Données et Regarde-les")
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

            # Fonctionnalités d'Analyse Supplémentaires
            st.subheader("Explore tes Données Plus Loin :")

            if numeric_columns:
                # Calculs Simples
                st.subheader("Calculs Simples :")
                selected_calculation_column = st.selectbox("Choisis une colonne pour les calculs :", numeric_columns)
                if selected_calculation_column:
                    st.write(f"Moyenne de {selected_calculation_column}: {data[selected_calculation_column].mean():.2f}")
                    st.write(f"Médiane de {selected_calculation_column}: {data[selected_calculation_column].median():.2f}")
                    st.write(f"Minimum de {selected_calculation_column}: {data[selected_calculation_column].min()}")
                    st.write(f"Maximum de {selected_calculation_column}: {data[selected_calculation_column].max()}")
                    st.write(f"Somme de {selected_calculation_column}: {data[selected_calculation_column].sum():.2f}")

                # Pourcentages (si au moins une colonne non numérique existe)
                non_numeric_columns = data.select_dtypes(exclude=["number"]).columns.tolist()
                if non_numeric_columns:
                    st.subheader("Pourcentages :")
                    selected_percentage_column = st.selectbox("Choisis une colonne pour les pourcentages :", non_numeric_columns)
                    if selected_percentage_column:
                        total_count = len(data)
                        value_counts = data[selected_percentage_column].value_counts()
                        percentages = (value_counts / total_count) * 100
                        st.write(percentages.rename("Pourcentage"))

                # Visualisation des Données (Nuage de Points)
                st.subheader("Nuage de Points :")
                if len(numeric_columns) >= 2:
                    x_column = st.selectbox("Choisis la colonne pour l'axe des X :", numeric_columns)
                    y_column = st.selectbox("Choisis la colonne pour l'axe des Y :", numeric_columns)
                    if x_column and y_column:
                        scatter_chart = alt.Chart(data).mark_circle().encode(
                            x=x_column,
                            y=y_column,
                            tooltip=[x_column, y_column]
                        ).properties(width=600, height=400).interactive()
                        st.altair_chart(scatter_chart, use_container_width=True)
                else:
                    st.info("Il faut au moins deux colonnes numériques pour faire un nuage de points.")

                # Visualisation des Données (Boîtes à Moustaches)
                st.subheader("Boîtes à Moustaches :")
                if len(numeric_columns) >= 1 and len(non_numeric_columns) >= 1:
                    boxplot_y_column = st.selectbox("Choisis la colonne numérique :", numeric_columns)
                    boxplot_x_column = st.selectbox("Choisis la colonne de catégories :", non_numeric_columns)
                    if boxplot_y_column and boxplot_x_column:
                        boxplot_chart = alt.Chart(data).mark_boxplot().encode(
                            x=boxplot_x_column,
                            y=boxplot_y_column,
                            tooltip=[boxplot_x_column, boxplot_y_column]
                        ).properties(width=600, height=400).interactive()
                        st.altair_chart(boxplot_chart, use_container_width=True)
                else:
                    st.info("Il faut au moins une colonne numérique et une colonne de catégories pour faire une boîte à moustaches.")

            else:
                st.info("Il faut des colonnes avec des chiffres pour faire des analyses.")

            # Filtres
            st.subheader("Filtrer tes Données :")
            for col in data.columns:
                unique_values = data[col].unique()
                if len(unique_values) < 50:  # Pour ne pas afficher trop de valeurs
                    selected_values = st.multiselect(f"Filtrer par {col}:", unique_values)
                    if selected_values:
                        data = data[data[col].isin(selected_values)]
                        st.subheader("Données Filtrées :")
                        st.dataframe(data.head())

            # Tri
            st.subheader("Trier tes Données :")
            sort_column = st.selectbox("Choisir une colonne pour trier :", data.columns)
            ascending = st.checkbox("Trier du plus petit au plus grand", True)
            if sort_column:
                sorted_data = data.sort_values(by=sort_column, ascending=ascending)
                st.subheader("Données Triées :")
                st.dataframe(sorted_data.head())

            # Sélection de Colonnes pour l'Aperçu
            st.subheader("Choisir les Colonnes à Afficher :")
            columns_to_show = st.multiselect("Sélectionne les colonnes :", data.columns, default=data.columns.tolist())
            if columns_to_show:
                st.subheader("Aperçu des Colonnes Sélectionnées :")
                st.dataframe(data[columns_to_show].head())

            # Aide de l'IA (Marina AI)
            st.header("Pose tes Questions à Marina AI :")
            st.write("Exemples de questions :")
            st.write("- Quelle est la moyenne de [Nom de la colonne] ?")
            st.write("- Y a-t-il une relation entre [Colonne A] et [Colonne B] ?")
            st.write("- Montre-moi les données pour [Catégorie spécifique] dans [Nom de la colonne].")
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
                                "max_tokens": 100, # Augmenter un peu pour les réponses plus complexes
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
