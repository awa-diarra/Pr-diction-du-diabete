# Prédiction du diabète

## Description
Projet de Machine Learning comparant trois algorithmes de classification pour prédire le diabète chez des patients à partir de données médicales. Ce projet compare trois modèles : **Decision Tree, Random Forest et XGBoost**.

## Objectif
Construire un modèle de classification capable de détecter 
les patients diabétiques avec le meilleur Recall possible,
car en médecine, rater un vrai diabétique est très dangereux.

## Dataset
**Pima Indians Diabetes Dataset**
- 768 patients
- 8 features médicales
- Cible: Outcome ( 0 = sain, 1 = diabétique)

## Features utilisés
- Glucose: Taux de glucose;
- Insulin: Taux d'insuline;
- BMI: Indice de masse corporelle;
- Age: Age du patient;
- Pregnancies: Nombre de grossesses;
- BloodPressure: Pression artérielle;
- SkinThickness: Épaisseur de la peau;
- DiabetesPedigreeFunction: Antécédents familiaux

## Résultats des modèles testés
- Decision Tree (base): Accuracy: 80.7% | Recall: 78.3%
- Decision Tree (optimisé): Accuracy: 82.8% | Recall: 82.6%
- Random Forest: Accuracy (base) : 89.1% | Recall: 85.5%
- Random Forest (optimisé) : Accuracy: 89.1% | Recall: 87.0%
- XGBoost (base): Accuracy: 87.5% | Recall: 84.1%
- XGBoost (optimisé) : Accuracy: 85.9% | Recall: 89.9%
- XGBoost (cross-validation cv=5) : Accuracy: 88.0% | Recall: 88.1%

## Technologies utilisées
- Python
- Pandas
- Scikit-learn
- Streamlit
- Kaggle Notebooks
- Visual Studio Code

## Modèle final - XGBoost
Le XGBoost (optimisé) a été retenu car il obtient le meilleur Recall (89.9%), métrique en contexte médical.

**Paramètres optimaux:**
```python
XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=4 ,scale_pos_weight=2, random_state=1)
```

## Pourquoi le Recall?
En médecine, rater un vrai diabétique est plus dangereux qu'une fausse alarme. Le Recall mesure le pourcentage de vrais diabétiques détectés par le modèle.

## Structure du projet
|-- notebook.ipynb : Analyse complète

|-- app.py : Interface Web Streamlit

|-- diabetes.csv : Dataset

|-- requirements.txt

## Notebook Kaggle
  [Voir le notebook complet](https://www.kaggle.com/code/diarraawa06/pr-diction-du-diab-te)

## Interface web
Application développée avec Streamlit permettant de prédire le risque de diabète en entrant les données médicales d'un patient.

## Pistes d'amélioration
- Ajuster le seuil de décision (50% à 30%)
- Enrichir le dataset avec plus de patients
- Tester LightGBM et les réseaux de neurones

## Auteure
**Awa DIARRA** Etudiante Ingénieur Informatique EILCO.
- LinkedIn : (www.linkedin.com/in/awa-diarra-112564358)
- Kaggle : (https://www.kaggle.com/diarraawa06)
- GitHub : (https://github.com/Awa-Diarra)
