# Classification-des-tweets
Le projet vise à classifier des tweets en "suspects" et "non suspects" en utilisant des données équilibrées pour éviter les biais. Le prétraitement est réalisé avec SpaCy. Le sous-échantillonnage gère le déséquilibre des classes.  Logistic regression, random et SVM ont été pour la prédiction.
### Prérequis
Pour exécuter ce projet, assurez-vous d'avoir installé les dépendances suivantes :
Python 3.x, Pandas, Matplotlib, Seaborn, spaCy, scikit-learn.
### 1. Structure du Code
#### 1.1. Importation des Bibliothèques
Dans cette section, nous importons toutes les bibliothèques nécessaires pour la manipulation des données, la visualisation, et les algorithmes d'apprentissage automatique.
#### 1.2. Chargement et Exploration des Données
Cette section charge le fichier des tweets suspects(voir dossier dataset), vérifie la structure des données, et effectue quelques visualisations exploratoires pour comprendre les classes des tweets.
##### Equilibrage des données
Le sous echantillonnage est utilisé qpour ré-equilibré des données
#### 1.3. Prétraitement des Données

- La suppression des caractères spéciaux avec spaCy
- La tokenisation avec spaCy
- La vectorisation des données textuelles avec TF-IDF

#### 1.4. Division des Données
Utilisation de train_test_split pour séparer les données en ensembles d'entraînement et de test.
#### 1.5. Entraînement du Modèle
Application des algorithmes de classification et comparaison de leurs performances :

- **SVM**
- **Random Forest**
- **K-Nearest Neighbors**, etc.
#### 1.6. Ajustement des hyperparamètres 
Application de ces algorithmes de classification avec les hyperparamètres et comparaison de leurs performances
#### 1.7. Évaluation du Modèle
Présentation des métriques de performance, telles que l'accuracy, la précision, les matrices de confusion et le rappel, pour évaluer la qualité des modèles.
### 2. Streamlit
- **run_streamlit.py** : Script principal pour exécuter l'application Streamlit. Ce fichier lance l'interface utilisateur, permettant de charger et tester les modèles de classification (régression logistique, forêt aléatoire, SVM) sur des tweets pour déterminer s'ils sont suspects ou non.

- **logistic_regression_model.pkl** : Modèle de régression logistique pré-entraîné, enregistré au format `.pkl`. Ce modèle est utilisé pour la classification des tweets en "suspects" ou "non suspects" sur la base des caractéristiques textuelles extraites.

- **random_forest_model.pkl** : Modèle de forêt aléatoire pré-entraîné, également enregistré au format `.pkl`. Ce modèle applique une approche d'ensemble d'arbres de décision pour améliorer la précision de la classification des tweets.

- **vm_model.pkl** : Modèle de Support Vector Machine (SVM) pré-entraîné, enregistré dans un fichier `.pkl`. Ce modèle est utilisé pour classifier les tweets en maximisant la séparation entre les classes à l'aide de vecteurs de support.

### 3. Figures
