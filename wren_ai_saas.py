import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import requests
import json

# Clé API Mistral
MISTRAL_API_KEY = "0AHCDGLvx7MFocAFM9Pw8lMAoFhZyNvu"

# Style CSS pour la barre latérale (amélioré)
sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-color: #f9faff; /* Fond légèrement plus chaud */
    padding: 20px;
    border-right: 1px solid #e6e6e6;
    box-shadow: 2px 0px 5px rgba(0, 0, 0, 0.05); /* Ombre légère */
}
.sidebar-title {
    color: #2e8b57; /* Vert forêt */
    font-size: 2em;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 3px solid #2e8b57;
}
.file-uploader {
    margin-bottom: 20px;
    padding: 15px;
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    background-color: #fff;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.03);
}
.st-expander {
    border: 1px solid #dcdcdc;
    border-radius: 8px;
    margin-bottom: 15px;
    background-color: #fff;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.03);
}
.st-expander-header {
    font-weight: bold;
    color: #333;
    padding: 12px;
}
.prediction-section {
    padding: 15px;
    background-color: #f0fff0; /* Vert très clair */
    border-radius: 5px;
    margin-top: 10px;
}
.prediction-subheader {
    color: #1e90ff; /* Bleu vif */
    font-weight: bold;
    margin-bottom: 8px;
}
.stButton > button:first-child {
    background-color: #2e8b57; /* Vert forêt */
    color: white;
    border-radius: 10px;
    height: 3.5em;
    width: 100%;
    margin-top: 15px;
    box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease;
}
.stButton > button:first-child:hover {
    background-color: #3cb371; /* Vert plus clair au survol */
}
.analysis-option {
    padding: 8px 0;
    border-bottom: 1px solid #eee;
}
.analysis-option:last-child {
    border-bottom: none;
}
.analysis-title {
    font-weight: bold;
    color: #555;
}
.analysis-description {
    color: #777;
    font-size: 0.9em;
}
.st-info {
    background-color: #fffacd; /* Jaune pâle */
    color: #000000; /* Noir */
    padding: 15px;
    border-radius: 5px;
    margin-top: 10px;
    border: 1px solid #eee8aa; /* Bordure jaune plus foncée (optionnel) */
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Style CSS principal
custom_css = """
<style>
h1 {
    color: #2e8b57;
    font-size: 2.8em;
    margin-bottom: 15px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar (fixed "Analyses Supplémentaires")
st.sidebar.markdown('<p class="sidebar-title">🔧 Paramètres</p>', unsafe_allow_html=True)
uploaded_file_sidebar = st.sidebar.file_uploader("📁 Téléchargez votre fichier CSV ou Excel", type=["csv", "xlsx", "xls"], key="file_uploader")

st.sidebar.markdown('<div class="st-expander" style="border: none; background-color: transparent; box-shadow: none;">'
                    '<div class="st-expander-header" style="font-weight: bold; color: #333; padding: 12px;">'
                    '⚙️ Analyses Supplémentaires</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="padding: 10px;">'
                    '<p class="analysis-title">📊 Visualisations</p>'
                    '<p class="analysis-description">Explorez vos données à travers des graphiques interactifs.</p>'
                    '</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="padding: 10px;">'
                    '<p class="analysis-title">🌍 Comparaison Géographique</p>'
                    '<p class="analysis-description">Comparez des métriques entre différentes catégories (ex: pays).</p>'
                    '</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div style="padding: 10px;">'
                    '<p class="analysis-title">🕒 Analyse Temporelle</p>'
                    '<p class="analysis-description">Visualisez les tendances de vos données au fil du temps.</p>'
                    '</div>', unsafe_allow_html=True)

if uploaded_file_sidebar and 'data' in locals() and data is not None and not data.empty:
    st.sidebar.markdown('<div class="prediction-section">'
                        '<p class="prediction-subheader">🔮 Prédictions Simples</p>'
                        '</div>', unsafe_allow_html=True)
    predict_date_col = st.sidebar.selectbox("📅 Colonne de date", data.columns, key="date_col")
    predict_numeric_col = st.sidebar.selectbox("🔢 Colonne numérique", data.select_dtypes(include=['number']).columns, key="numeric_col")
    predict_button = st.sidebar.button("🚀 Lancer la Prédiction", key="predict_button")

st.sidebar.markdown('</div>', unsafe_allow_html=True) # Fermeture du div pour l'expander (simulé)

# Titre
st.title("Expert en Données : Interrogez vos Données et réduisez vos dépenses !")
st.write("""
Cette plateforme t'aide à comprendre tes données et à résoudre des problèmes.
- **Téléchargez vos données** (CSV ou Excel)
- **Visualisez et nettoyez vos données**
- **Interrogez-les avec l'IA** (locale et potentiellement en ligne)
""")

uploaded_file = uploaded_file_sidebar
data = None

if uploaded_file:
    try:
        ext = os.path.splitext(uploaded_file.name)[1]
        if ext in [".xlsx", ".xls"]:
            data = pd.read_excel(uploaded_file)
        else:
            data = pd.read_csv(uploaded_file, on_bad_lines='skip')

        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data.columns = [col.encode('ascii', 'ignore').decode().replace(':', '').strip() for col in data.columns]

        st.toast("Fichier chargé avec succès ✅", icon="📥")
    except Exception as e:
        st.error(f"❌ Erreur lors de chargement du fichier : {e}")
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

    for col in data.columns:
        if data[col].dtype == 'object':
            cleaned_col = pd.to_numeric(data[col], errors='coerce')
            if cleaned_col.notna().sum() > 0:
                data[col] = cleaned_col

    datetime_cols = []
    for col in data.columns:
        if data[col].dtype == 'object' or 'time' in col.lower() or 'date' in col.lower():
            try:
                converted_col = pd.to_datetime(data[col], errors='coerce', infer_datetime_format=True)
                if converted_col.notna().sum() > 0:
                    data[col] = converted_col
                    datetime_cols.append(col)
            except Exception:
                continue

    if datetime_cols:
        st.subheader("🕒 Analyse Temporelle")
        time_col = st.selectbox("🗓️ Colonne de temps", datetime_cols)
        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            metric_col = st.selectbox("📈 Métrique à afficher", numeric_cols)
            line_chart = alt.Chart(data).mark_line().encode(
                x=alt.X(time_col, title="Temps"),
                y=alt.Y(metric_col, title=metric_col),
                tooltip=[time_col, metric_col]
            ).interactive()
            st.altair_chart(line_chart, use_container_width=True)

    st.header("🌍 Comparaison entre Deux Pays")
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

    if categorical_cols and numeric_cols:
        group_col = st.selectbox("🧩 Colonne catégorielle représentant les pays", categorical_cols)
        metric_col = st.selectbox("📊 Métrique numérique à comparer", numeric_cols, key="metric")

        unique_groups = data[group_col].dropna().unique().tolist()
        if len(unique_groups) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                selected_country1 = st.selectbox("🌐 Pays 1", unique_groups, key="country1")
            with col2:
                selected_country2 = st.selectbox("🌐 Pays 2", unique_groups, key="country2")

            subset = data[data[group_col].isin([selected_country1, selected_country2])]
            comparison_chart = alt.Chart(subset).mark_bar().encode(
                x=alt.X(group_col, sort='-y'),
                y=alt.Y(metric_col),
                color=group_col,
                tooltip=[group_col, metric_col]
            ).interactive()
            st.altair_chart(comparison_chart, use_container_width=True)

    st.header("📈 Visualisation personnalisée")
    st.write("Choisissez l'indicateur que vous souhaitez visualiser")
    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    selected_metric = st.selectbox("📌 Sélectionnez une métrique", numeric_columns)

    if selected_metric:
        st.markdown(f"### 📊 Histogramme de {selected_metric}")
        hist_chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(selected_metric, bin=alt.Bin(maxbins=30)),
            y='count()'
        ).interactive()
        st.altair_chart(hist_chart, use_container_width=True)

    if len(numeric_columns) >= 2:
        st.markdown("### 📌 Nuage de points (Scatter plot)")
        x_col = st.selectbox("🟦 Axe X", numeric_columns, key="x")
        y_col = st.selectbox("🟥 Axe Y", numeric_columns, key="y")
        scatter_chart = alt.Chart(data).mark_circle(size=60).encode(
            x=x_col, y=y_col, tooltip=[x_col, y_col]
        ).interactive()
        st.altair_chart(scatter_chart, use_container_width=True)

    st.subheader("🔗 Corrélation entre variables")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    st.subheader("💾 Télécharger les Données Nettoyées")
    st.info(f"- {rows_dropped_na} lignes avec valeurs manquantes supprimées\n- {rows_dropped_duplicates} doublons supprimées")
    st.download_button("📥 Télécharger les données nettoyées", data.to_csv(index=False), "donnees_nettoyees.csv", key="download_button")

    # Résumé IA
    if st.checkbox("🧠 Obtenir un résumé automatique des données", key="summary_checkbox"):
        if MISTRAL_API_KEY:
            with st.spinner("Génération du résumé avec Mistral AI..."):
                sample_records = data.sample(min(len(data), 50), random_state=42).astype(str).to_dict(orient='records')
                sample_json = json.dumps(sample_records, ensure_ascii=False)
                prompt = f"Voici un échantillon de données :\n{sample_json}\nFais un résumé clair des types, tendances et anomalies potentielles. N'invente rien."
                headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "mistral-tiny",
                    "messages": [
                        {"role": "system", "content": "Vous êtes un analyste de données expérimenté."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300
                }
                try:
                    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    response_json = response.json()
                    if "choices" in response_json:
                        summary = response_json["choices"][0]["message"]["content"]
                        st.markdown("### 📾 Résumé IA (Mistral AI)")
                        st.write(summary)
                    else:
                        st.error("La réponse de l'API ne contient pas de résultat.")
                        st.json(response_json)
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur API Mistral : {e}. Utilisation du résumé local.")
                    st.write("Colonnes numériques :", data.select_dtypes(include=["number"]).columns.tolist())
                    st.write("Colonnes catégorielles :", data.select_dtypes(include=["object", "category"]).columns.tolist())
                    st.write(data.describe())

    question = st.text_input("❓ Posez une question à l'IA", key="question_input")
    if question:
        if MISTRAL_API_KEY:
            with st.spinner("Recherche de la réponse avec Mistral AI..."):
                sample_records = data.sample(min(len(data), 50), random_state=42).astype(str).to_dict(orient='records')
                sample_json = json.dumps(sample_records, ensure_ascii=False)
                prompt = (
                    f"Voici un échantillon de données :\n{sample_json}\n"
                    f"Question : {question}\n"
                    f"Réponds de manière claire et directe, avec des chiffres clés si possible. "
                    f"N'invente pas de données. Base ta réponse uniquement sur l’échantillon fourni."
                )
                headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
                payload = {
                    "model": "mistral-tiny","messages": [
                        {"role": "system", "content": "Vous êtes un analyste de données expérimenté."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 300
                }
                try:
                    response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers, timeout=10)
                    response.raise_for_status()
                    response_json = response.json()
                    if "choices" in response_json:
                        answer = response_json["choices"][0]["message"]["content"]
                        st.markdown("### Réponse de l'IA")
                        st.write(answer)
                    else:
                        st.error("La réponse de l'API ne contient pas de résultat.")
                        st.json(response_json)
                except requests.exceptions.RequestException as e:
                    st.error(f"Erreur API Mistral : {e}")
        else:
            st.warning("🛠️ Clé API Mistral manquante, incapable d'interroger l'IA.")
else:
    st.info("""
🎯 **Commencez votre analyse maintenant !**

Téléchargez votre fichier **CSV ou Excel** pour :
- Comprendre vos données
- Identifier des leviers d’économies
- Poser des questions à l’IA
""")
