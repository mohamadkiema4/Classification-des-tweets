# Classification-des-tweets
Le projet vise à classifier des tweets en "suspects" et "non suspects" en utilisant des données équilibrées pour éviter les biais. Le prétraitement est réalisé avec SpaCy. Le sous-échantillonnage gère le déséquilibre des classes.  Logistic regression, random et SVM ont été pour la prédiction.
### Prérequis
Pour exécuter ce projet, assurez-vous d'avoir installé les dépendances suivantes :
Python 3.x, Pandas, Matplotlib, Seaborn, spaCy, scikit-learn.
### Structure du Code
#### 1. Importation des Bibliothèques
Dans cette section, nous importons toutes les bibliothèques nécessaires pour la manipulation des données, la visualisation, et les algorithmes d'apprentissage automatique.
#### 2. Chargement et Exploration des Données
Cette section charge le fichier des tweets suspects(voir dossier dataset), vérifie la structure des données, et effectue quelques visualisations exploratoires pour comprendre les classes des tweets.
##### Equilibrage des données
Le sous echantillonnage est utilisé qpour ré-equilibré des données
#### 3. Prétraitement des Données
        -La suppression des caractères spéciaux avec spaCy
        - La tokenisation avec spaCy
        -La vectorisation des données textuelles avec TF-IDF
#### 4. Division des Données
Utilisation de train_test_split pour séparer les données en ensembles d'entraînement et de test.
#### 5. Entraînement du Modèle
Application des algorithmes de classification et comparaison de leurs performances :
        *SVM
        *Random Forest
        *K-Nearest Neighbors, etc.
#### Ajustement des hyperparamètres 
Application de ces algorithmes de classification avec les hyperparamètres et comparaison de leurs performances
#### 6. Évaluation du Modèle
Présentation des métriques de performance, telles que l'accuracy, la précision, les matrices de confusion et le rappel, pour évaluer la qualité des modèles.
### Streamlit
### Figures
