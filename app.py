#importer les bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.markdown("""
    <style>
    /* Arrière plan principal */
    .stApp {
        background-color: #fdf6f0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fde8d8;
    }
    
    /* Titre principal */
    h1 {
        color: #c0737a;
        font-family: Georgia, serif;
    }
    
    /* Sous-titres */
    h2, h3 {
        color: #d4956a;
    }
    
    /* Bouton */
    .stButton > button {
        background-color: #f4a5a5;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 30px;
        font-size: 16px;
    }
    
    /* Bouton au survol */
    .stButton > button:hover {
        background-color: #e07b7b;
        color: white;
    }
    
    /* Messages */
    .stSuccess {
        background-color: #d4edda;
    }
    
    .stError {
        background-color: #f8d7da;
    }
    </style>
""", unsafe_allow_html=True)



# Titre et description
st.title("Prédiction du Diabète")
st.sidebar.info("Remplissez vos données médicales et cliquez sur Prédire")

#charger et entrainer le modèle

@st.cache_resource
def charger_modele():
    donnee=pd.read_csv("diabetes.csv")
    y=donnee["Outcome"]
    X=donnee.drop(columns="Outcome")
    model=RandomForestClassifier(random_state=1)
    model.fit(X,y)
    return model

#interce utilisateur
st.sidebar.header("Entrez vos données:")

pregnancies = st.sidebar.slider("Nombre de grossesses", 0, 17, 3)
glucose = st.sidebar.slider("Taux de glucose", 0, 200, 120)
blood_pressure = st.sidebar.slider("Pression artérielle", 0, 122, 70)
skin_thickness = st.sidebar.slider("Épaisseur de la peau", 0, 99, 20)
insulin = st.sidebar.slider("Taux d'insuline", 0, 846, 80)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.slider("Antécédents familiaux (DPF)", 0.0, 2.4, 0.5)
age = st.sidebar.slider("Âge", 21, 81, 30)

#entré de prédiction
data=np.array([[pregnancies,glucose,blood_pressure,
               skin_thickness,insulin,bmi,dpf,
               age]])
#bouton de prediction



if st.button("Prédire"):
    modele=charger_modele()
    prediction=modele.predict_proba(data)
    
    # Barre de progression
    st.progress(float(prediction[0][1]))
    st.write("Chance d'être diabétique:"+str(round(prediction[0][1]*100,2))+"%")


    if prediction[0][1] >0.65:
        st.error("Attention! Ce patient a de fortes chances d'être diabétique")
    elif prediction[0][1] >0.35:
        st.warning("Faites des tests à ce patient par mesure de prévention")
    else:
        st.success("Patient a de faibles risques de diabètes")

# Créer un DataFrame avec les valeurs entrées
input_df = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [dpf],
    'Age': [age]
})

# Afficher le tableau
st.subheader("Vos données")
st.dataframe(input_df,hide_index=True)