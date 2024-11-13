import streamlit as st
import spacy
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#------------------------------------------------------------------------------------#

st.markdown(
    """
    <style>
    .stApp {
        background-color: #33ffff
    }
    </style>
    """,
    unsafe_allow_html=True
)

#------------------------------------------------------------------------------------#
# Charger le dataset
df = pd.read_csv('tweets_suspect.csv')  # fichier contenant les tweets suspects
# Charger les modèles entraînés
log_reg_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
# Charger le TF-IDF existant
tfidf = joblib.load('tfidf_vectorizer.pkl')
# Charger le modèle SpaCy pour le nettoyage de texte
nlp = spacy.load("en_core_web_sm")

# Fonction pour nettoyer le texte avec SpaCy
def nettoyer_texte_spacy(texte):
    doc = nlp(texte.lower())  # Convertir en minuscules et traiter avec SpaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# Interface utilisateur avec Streamlit
st.title("Classification de Tweets Suspects")

st.sidebar.title("Sommaire")
pages = ["Contexte du projet", "Exploration des données", "Analyse de tweets"]
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0]: 
    st.write("### Contexte du projet")
    st.write("Ce projet exploite des techniques de traitement du langage naturel et d'apprentissage automatique, notamment la régression logistique, la forêt aléatoire et les machines à vecteurs de support, pour classifier les tweets en tant que suspects ou non suspects.")
    st.image("imag.jpg", width=800)

elif page == pages[1]:
    st.write("### Exploration des données")
    st.dataframe(df.head())
    st.write("Dimensions du dataframe : ", df.shape)
    
    st.write("#### Analyse univariée des données")
    label_distribution = df['label'].value_counts()
    colors = ['red', 'green']
    
    # Barplot de la distribution des labels
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(label_distribution.index, label_distribution.values, color=colors)
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution des étiquettes dans les données')
    ax1.set_xticks([1, 0])
    ax1.set_xticklabels(['Suspect (1)', 'Non-Suspect (0)'])
    st.pyplot(fig1)
    
    # Pie chart de la distribution des labels
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.pie(label_distribution.values, labels=['Suspect (1)', 'Non-Suspect (0)'], autopct='%1.1f%%', colors=colors)
    ax2.set_title('Proportion des étiquettes dans les données')
    st.pyplot(fig2)

elif page == pages[2]: 
    st.write("#### Entrez un texte pour déterminer s'il est suspect ou non.")
    
    message = st.text_area("###### Entrez le texte ici", "")
    
    # Sélection du modèle
    model_choice = st.selectbox(
        "###### Choisissez un modèle pour la prédiction",
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    if st.button("Analyser et Prédire"):
        if message:
            # Nettoyage du texte
            message_nettoye = nettoyer_texte_spacy(message)
            if not message_nettoye:
                st.warning("Le texte nettoyé est vide. Veuillez entrer un texte plus pertinent.")
            else:
                # Vectorisation avec le modèle TF-IDF chargé
                #tfidf = TfidfVectorizer(max_features=5000)
                #message_vect = tfidf([message_nettoye]).toarray()
                # Appliquer le TF-IDF vectorizer chargé pour transformer le texte nettoyé
                message_vect = tfidf.transform([message_nettoye]).toarray()

                # Prédiction
                try:
                    if model_choice == 'Logistic Regression':
                        y_pred = log_reg_model.predict(message_vect)
                    elif model_choice == 'Random Forest':
                        y_pred = rf_model.predict(message_vect)
                    elif model_choice == 'SVM':
                        y_pred = svm_model.predict(message_vect)
                    
                    st.write(f"Résultat de la prédiction : {'Suspect' if y_pred[0] == 1 else 'Non-Suspect'}")
                
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {e}")
