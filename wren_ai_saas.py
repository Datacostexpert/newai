import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import requests
import json

# Remplacez 'YOUR_API_KEY' par votre clé API Mistral (si vous souhaitez utiliser l'IA en ligne)
MISTRAL_API_KEY = "0AHCDGLvx7MFocAFM9Pw8lMAoFhZyNvu"  # Gardez cette ligne pour une utilisation future de l'API Mistral

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

# Barre latérale
with st.sidebar:
    st.title("🔧 Paramètres")
    uploaded_file_sidebar = st.file_uploader("📁 Téléchargez votre fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
    st.markdown("### ⚙️ Analyses Supplémentaires")
    st.markdown("### 🔮 Prédictions Simples")
    st.caption("Besoin d'une colonne de date et d'une autre numérique pour la prédiction.")

# Titre principal
st.title("Expert en Données : Interrogez vos Données et réduisez vos dépenses !")
st.write("""
Cette plateforme t'aide à comprendre tes données et à résoudre des problèmes.
- **Téléchargez vos données** (CSV ou Excel)
- **Visualisez et nettoyez vos données**
- **Interrogez-les avec l'IA** (locale et potentiellement en ligne)
""")

# Fichier Upload principal
uploaded_file_main = st.file_uploader("Ou téléchargez un fichier ici", type=["csv", "xlsx", "xls"])
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

if data is not None and not data.empty:
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
            data[col] = pd.to_numeric(data[col], errors='coerce')
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Résumé IA (priorité à l'option en ligne, avec fallback local)
    if st.checkbox("🧠 Obtenir un résumé automatique des données"):
        if MISTRAL_API_KEY:  # Vérifie si la clé API est fournie
            with st.spinner("Génération du résumé avec Mistral AI..."):
                sample_json = data.head(10).to_json(orient='records')
                prompt = f"Voici un échantillon des données : {sample_json}. Fais un résumé rapide : types, tendances, anomalies."
                headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}",
                           "Content-Type": "application/json"}
                payload = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system",
                         "content": "Vous êtes un analyste de données expérimenté."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300
                }
                try:
                    response = requests.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=10  # Ajoute un timeout pour éviter les blocages
                    )
                    response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
                    summary = response.json()["choices"][0]["message"]["content"]
                    st.markdown("### 📾 Résumé IA (Mistral AI)")
                    st.write(summary)
                except requests.exceptions.RequestException as e:
                    st.error(
                        f"Erreur lors de l'appel à l'API Mistral : {e}.  Utilisation du résumé local.")
                    # Fallback au résumé local
                    st.markdown("### 📾 Résumé IA (Local)")
                    st.markdown("#### 🔢 Types de colonnes")
                    st.write("- Colonnes numériques :", numeric_columns)
                    st.write("- Colonnes catégorielles :", categorical_columns)
                    st.markdown("#### 🧪 Statistiques principales")
                    st.write(data.describe())
                    if data.isnull().sum().sum() > 0:
                        st.warning("Des valeurs manquantes ont été détectées.")
                    if len(data.drop_duplicates()) < len(data):
                        st.info("Des doublons ont été supprimés.")
                    st.markdown("#### 💡 Suggestions de base")
                    st.markdown(
                        "- Vérifiez les colonnes ayant peu de valeurs uniques.")
                    st.markdown(
                        "- Analysez les colonnes avec de fortes corrélations.")
        else:
            # Si la clé API n'est pas fournie, utilisez le résumé local
            st.markdown("### 📾 Résumé IA (Local)")
            st.markdown("#### 🔢 Types de colonnes")
            st.write("- Colonnes numériques :", numeric_columns)
            st.write("- Colonnes catégorielles :", categorical_columns)
            st.markdown("#### 🧪 Statistiques principales")
            st.write(data.describe())
            if data.isnull().sum().sum() > 0:
                st.warning("Des valeurs manquantes ont été détectées.")
            if len(data.drop_duplicates()) < len(data):
                st.info("Des doublons ont été supprimés.")
            st.markdown("#### 💡 Suggestions de base")
            st.markdown("- Vérifiez les colonnes ayant peu de valeurs uniques.")
            st.markdown("- Analysez les colonnes avec de fortes corrélations.")

    st.subheader("🔍 Types de Données")
    st.write(data.dtypes)

    st.subheader("❓ Valeurs Manquantes")
    st.write(data.isnull().sum())

    st.subheader("📊 Statistiques Descriptives")
    stats_description = data.describe()
    st.write(stats_description)

    st.header("📈 Visualisation des Données")

    if numeric_columns:
        selected_column = st.selectbox("📌 Colonne pour histogramme",
                                     numeric_columns)
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
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm',
                    fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Aucune colonne numérique détectée.")

    if categorical_columns:
        selected_cat = st.selectbox("Colonne catégorielle", categorical_columns)
        st.bar_chart(data[selected_cat].value_counts())

    st.subheader("💾 Télécharger les Données Nettoyées")
    cleaning_summary = ""
    if rows_dropped_na > 0:
        cleaning_summary += (
            f"- {rows_dropped_na} lignes avec valeurs manquantes supprimées\n")
    if rows_dropped_duplicates > 0:
        cleaning_summary += (
            f"- {rows_dropped_duplicates} doublons supprimés\n")
    cleaning_summary += "- Tentative de conversion des textes en chiffres\n"
    st.info(cleaning_summary)
    st.download_button("📥 Télécharger les données nettoyées",
                       data.to_csv(index=False), "donnees_nettoyees.csv")

    st.header("🎯 Importance des variables")
    importances = None
    if len(categorical_columns) > 0:
        target_column = st.selectbox(
            "Sélectionnez la colonne cible (à prédire)", categorical_columns)
        if target_column:
            df_model = data.copy()
            df_model = df_model.dropna()
            label_encoders = {}
            for col in df_model.columns:
                if df_model[col].dtype == 'object' and col != target_column:
                    le = LabelEncoder()
                    df_model[col] = le.fit_transform(
                        df_model[col].astype(str))
                    label_encoders[col] = le
            y = df_model[target_column]
            X = df_model.drop(columns=[target_column])
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            model = RandomForestClassifier()
            model.fit(X, y_encoded)
            importances = pd.Series(model.feature_importances_,
                                    index=X.columns).sort_values(
                ascending=False)
            st.bar_chart(importances)

    # Assistance IA pour poser des questions sur les données
    st.header("🤖 Posez une question à l'IA sur vos données")
    user_question = st.text_input("Entrez votre question :")
    if st.button("Demander à l'IA"):
        if user_question:
            if MISTRAL_API_KEY:
                with st.spinner("Interrogation des données avec l'IA..."):
                    data_for_ai = data.head(100).to_json(
                        orient='records')  # Send only the first 100 rows
                    prompt = f"Tu es un expert en analyse de données. Réponds à la question suivante en utilisant uniquement les données fournies : {user_question}. Voici un échantillon des données au format JSON: {data_for_ai}.  Ne fais aucun calcul, donne la réponse directement en français.  Si la question ne peut pas être répondue avec les données fournies, réponds 'Je ne peux pas répondre à cette question avec les données fournies.'. Si la question porte sur le pays le plus riche, tu dois faire le calcul toi même à partir des données 'pib/h/20' et 'Population_totale' "
                    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}",
                               "Content-Type": "application/json"}
                    payload = {
                        "model": "mistral-tiny",
                        "messages": [
                            {"role": "system",
                             "content": "Vous êtes un expert en analyse de données."},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 300
                    }
                    try:
                        response = requests.post(
                            "https://api.mistral.ai/v1/chat/completions",
                            json=payload,
                            headers=headers,
                            timeout=20
                        )
                        response.raise_for_status()
                        ai_answer = response.json()["choices"][0]["message"][
                            "content"]
                        st.write(f"**Réponse de l'IA:** {ai_answer}")
                    except requests.exceptions.RequestException as e:
                        if e.response is not None and e.response.status_code == 401:
                            st.error(
                                "Erreur : Clé API Mistral invalide. Veuillez vérifier votre clé API et la mettre à jour.")
                        else:
                            st.error(
                                f"Erreur lors de l'interrogation des données avec l'IA : {e}")
            else:
                st.error(
                    "Veuillez entrer votre clé API Mistral pour utiliser cette fonctionnalité.")
        else:
            st.warning("Veuillez entrer une question.")

    # Génération de Rapport PDF
    if st.button("📄 Générer un Rapport PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Rapport Synthétique des Données", ln=True,
                 align="C")

        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        pdf.multi_cell(0, 10, "Résumé du nettoyage :\n" + cleaning_summary)
        pdf.ln(5)
        pdf.multi_cell(0, 10,
                       f"Nombre de lignes : {data.shape[0]}\nNombre de colonnes : {data.shape[1]}")
        pdf.ln(5)

        pdf.cell(0, 10, txt="Colonnes numériques :", ln=True)
        for col in numeric_columns:
            pdf.cell(0, 10, txt=f"- {col}", ln=True)
        pdf.ln(3)

        pdf.cell(0, 10, txt="Colonnes catégorielles :", ln=True)
        for col in categorical_columns:
            pdf.cell(0, 10, txt=f"- {col}", ln=True)

        if importances is not None:
            pdf.ln(5)
            pdf.cell(0, 10, txt="📊 Importance des variables (modèle interne) :",
                     ln=True)
            for feature, importance in importances.items():
                pdf.cell(0, 10, txt=f"{feature}: {importance:.2f}", ln=True)

        pdf.ln(5)
        # Utilisation de l'API Mistral pour le résumé, avec fallback local)
        if MISTRAL_API_KEY:
            try:
                sample_json = data.head(10).to_json(orient='records')
                prompt = "Donne un résumé concis des données, en mettant en évidence les principaux types de variables, les valeurs manquantes, les doublons, et les implications pour l'analyse. Sois bref, максимум 100 слов."
                headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}",
                           "Content-Type": "application/json"}
                payload = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system",
                         "content": "Tu es un analyste de données."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 150  # Limite la réponse pour éviter les réponses trop longues
                }
                response = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                response.raise_for_status()
                summary_text = response.json()["choices"][0]["message"][
                    "content"]
                pdf.multi_cell(0, 10,
                               "Résumé IA (Mistral):\n" + summary_text)
            except requests.exceptions.RequestException as e:
                pdf.multi_cell(0, 10,
                               f"Erreur lors de la génération du résumé IA : {e}. Utilisation du résumé local par défaut.")
                pdf.multi_cell(0, 10,
                               "Résumé IA (Local):\nLes données contiennent des colonnes numériques et catégorielles. Des valeurs manquantes et des doublons ont été traités. Les statistiques descriptives et les corrélations sont disponibles pour une analyse plus approfondie.")
        else:  # Si la clé n'est pas fournie
            pdf.multi_cell(0, 10,
                           "Résumé IA (Local):\nLes données contiennent des colonnes numériques et catégorielles. Des valeurs manquantes et des doublons ont été traités. Les statistiques descriptives et les corrélations sont disponibles pour une analyse plus approfondie.")

        filename = "rapport_automatique.pdf"
        pdf.output(filename)
        with open(filename, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f,
                               file_name=filename)
else:
    st.info("""
🎯 **Commencez votre analyse maintenant !**

Téléchargez votre fichier **CSV ou Excel** pour :
- Comprendre vos données
- Identifier des leviers d’économies
- Poser des questions à l’IA
- Générer un rapport PDF automatique

💡 Aucune compétence technique requise. Juste vos données.
""")
