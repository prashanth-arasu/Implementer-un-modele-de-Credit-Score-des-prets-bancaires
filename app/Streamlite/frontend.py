import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import image
import pickle
from urllib.request import urlopen
import json
import shap
import re

icon = image.imread("PaD.png")
st.title("Prêt à dépenser: Traitement des demandes du crédit")
st.image(icon, caption = 'Prêt à dépenser')
st.header("Bienvenue cher chargé du client")
st.markdown("> Choisir d'abord l'identifiant du client")

data = pd.read_csv("/app/Streamlite/app_tr.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data_client = pd.read_csv("/app/Streamlite/app_tr_all.csv")
accord = pd.read_csv("/app/Streamlite/app_tr_accord.csv")
refus = pd.read_csv("/app/Streamlite/app_tr_refus.csv")
feats = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
X = data.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
y = data['TARGET']
st.write(accord)
model = pickle.load(open('/app/FastAPI/Pickle_LGBM_Model.pkl', 'rb'))
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

data.drop(columns=['TARGET'], inplace=True)
#st.sidebar.title("Menu")
list = data['SK_ID_CURR']
ids = list.values
identifiant = st.selectbox('Choisir un Identifiant client: ', ids)
idx = data[data['SK_ID_CURR'] == identifiant].index.values
sidebar_selection = st.selectbox("Choisir l'option 'Lancer' afin d'accéder la prédiction", ['Rechoisir', 'Lancer'])
# affichage de la prédiction

if sidebar_selection == 'Lancer':
    st.write('## Décision sur la demande de crédit')
    with st.spinner('Calcul en cours'):
        API_url = "http://127.0.0.1:8000/api/clients/" + str(identifiant) + "/display"
    with st.spinner('Chargement des résultats...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        classe_predite = API_data['probability0']
        threshold = API_data['threshold']
        classe = float(classe_predite)
        if classe < 0.14:
            etat_0 = 'Client à risque. Crédit pas accordé'
            st.markdown("#Probabilité du repaiement:")
            st.write(API_data['probability0'])
            st.markdown('#Threshold:')
            st.markdown(threshold)
            st.markdown("#Décision sur la demande de crédit:")
            st.markdown(etat_0)
            decision = 0
        else:
            etat_1 = 'Client peu risqué. Crédit accordé'
            st.markdown("#Probabilité du repaiement:")
            st.write(API_data['probability0'])
            st.markdown('#Threshold:')
            st.markdown(threshold)
            st.markdown("#Décision sur la demande de crédit:")
            st.markdown(etat_1)
            decision = 1

# Feature importance / description
# shap.summary_plot(shap_values, features=df_clt, feature_names=df_clt.columns, plot_type='bar')

    st.write('## Interprétabilité du résultat')
    shap.initjs()
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][idx,:], X.iloc[idx,:]))

#shap.summary_plot(shap_values, features=X, plot_type="bar", max_display=10, color_bar=False, plot_size=(10, 10))

# affichage de la prédiction
    st.write('## Comparaison avec des clients similaires')
    comparaison_option = st.selectbox('Comparaison des clients', ('Source Externe 2', 'Enquetes de crédit',
                                                                  'Informations de contact laissé par le client',
                                                                  "Situation de famille:Marié(e) ou pas",
                                                                  "Employé en tant qu'ouvrier"))
    st.write('You selected:', comparaison_option)
    data_client = data_client[data_client['SK_ID_CURR'] == identifiant]
    if decision == 1:
        st.write('Comparaison avec des clients dont la demande de crédit a été accordé')
        if comparaison_option == "Source Externe 2":
            accord_df = accord['EXT_SOURCE_2']
            st.write("La valeur de la variable du client est: ", data_client['EXT_SOURCE_2'])
            st.bar_chart(accord_df)

        if comparaison_option == "Enquetes de crédit":
            accord_df = accord['Enquiry']
            st.write("La valeur de la variable du client est: ", data_client['Enquiry'])
            st.bar_chart(accord_df)

        if comparaison_option == "Informations de contact laissé par le client":
            accord_df = accord['Contact']
            st.write("La valeur de la variable du client est: ", data_client['Contact'])
            st.bar_chart(accord_df)

        if comparaison_option == "Situation de famille:Marié(e) ou pas":
            accord_df = accord['NAME_FAMILY_STATUS_Married']
            st.write("La valeur de la variable du client est: ", data_client['NAME_FAMILY_STATUS_Married'])
            st.bar_chart(accord_df)

        if comparaison_option == "Employé en tant qu'ouvrier":
            accord_df = accord['OCCUPATION_TYPE_Laborers']
            st.write("La valeur de la variable du client est: ", data_client['OCCUPATION_TYPE_Laborers'])
            st.bar_chart(accord_df)
    else:
        st.write('Comparaison avec des clients dont la demande de crédit a été refusé')
        if comparaison_option == "Source Externe 2":
            refus_df = refus['EXT_SOURCE_2']
            st.write("La valeur de la variable du client est: ", data_client['EXT_SOURCE_2'])
            st.bar_chart(refus_df)

        if comparaison_option == "Enquetes de crédit":
            refus_df = refus['Enquiry']
            st.write("La valeur de la variable du client est: ", data_client['Enquiry'])
            st.bar_chart(refus_df)

        if comparaison_option == "Informations de contact laissé par le client":
            refus_df = refus['Contact']
            st.write("La valeur de la variable du client est: ", data_client['Contact'])
            st.bar_chart(refus_df)

        if comparaison_option == "Situation de famille:Marié(e) ou pas":
            refus_df = refus['NAME_FAMILY_STATUS_Married']
            st.write("La valeur de la variable du client est: ", data_client['NAME_FAMILY_STATUS_Married'])
            st.bar_chart(refus_df)

        if comparaison_option == "Employé en tant qu'ouvrier":
            refus_df = refus['OCCUPATION_TYPE_Laborers']
            st.write("La valeur de la variable du client est: ", data_client['OCCUPATION_TYPE_Laborers'])
            st.bar_chart(refus_df)
    st.write('Note:')
    st.write('Index 0 represents the 0th Quartile among similar clients')
    st.write('Index 1 represents the 1st Quartile among similar clients')
    st.write('Index 2 represents the 2nd Quartile among similar clients')
    st.write('Index 3 represents the 3rd Quartile among similar clients')
    st.write('Index 4 represents the 4th Quartile among similar clients')