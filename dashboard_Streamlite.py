#Importation des libraries

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import image
import pickle
from urllib.request import urlopen
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Image et titre de la page

icon = image.imread("PaD.png")
st.title("Prêt à dépenser: Traitement des demandes du crédit")
st.image(icon)
st.header("Bienvenue cher chargé du client")
st.markdown("> Choisir d'abord l'identifiant du client")

# Importation des dataframes

data = pd.read_csv("app_tr.csv")
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
data_client = pd.read_csv("app_tr_all.csv")
accord = pd.read_csv("app_tr_accord.csv")
refus = pd.read_csv("app_tr_refus.csv")

# Préparation du dataframe

X = data.drop(['Unnamed0', 'SK_ID_CURR', 'TARGET'], axis = 1)
y = data['TARGET']

# Shap values

shap_val = pickle.load(open('shap_values.pkl', 'rb'))
exp_val = pickle.load(open('expected_shap_values.pkl', 'rb'))

# Button pour choisir l'ID du client

data.drop(columns=['TARGET'], inplace=True)
list = data['SK_ID_CURR']
ids = list.values
identifiant = st.selectbox('Choisir un Identifiant client: ', ids)
idx = data[data['SK_ID_CURR'] == identifiant].index.values
sidebar_selection = st.selectbox("Appuyer sur 'Lancer' afin d'accéder la prédiction", ['Rechoisir', 'Lancer'])

# Fonction pour la visualisation

def plot(df, col, val):
    fig = plt.figure(figsize=(10,5))
    graph = sns.barplot(x = df['Unnamed: 0'], y = df[col], data = df, alpha=0.8)
    graph.axhline(val, color = 'black', linestyle = '-.', markeredgewidth = 2.5)
    plt.title('Comparaison avec des client similaires')
    plt.ylabel(col, fontsize=12)
    plt.xlabel('Quartile', fontsize=12)
    return st.pyplot(fig)

# Affichage de la décision sur la demande de prêt d'argent obtenu de l'application FastAPI

if sidebar_selection == 'Lancer':
    st.write('## Décision sur la demande de crédit')
    with st.spinner('Calcul en cours'):
        API_url = "http://127.0.0.1:8000/api/clients/" + str(identifiant) + "/display"
    with st.spinner('Chargement des résultats...'):
        json_url = urlopen(API_url)
        API_data = json.loads(json_url.read())
        classe_predite = API_data['probability1']
        threshold = API_data['threshold']
        classe = float(classe_predite)
        if classe > 0.14:
            etat_0 = 'Client à risque. Crédit pas accordé'
            st.markdown(">Probabilité d'échéance du repaiement:")
            st.write(API_data['probability1'])
            st.markdown('>Seuil maximal:')
            st.markdown(threshold)
            st.markdown(">Décision sur la demande de crédit:")
            st.markdown(etat_0)
            decision = 0
        else:
            etat_1 = 'Client peu risqué. Crédit accordé'
            st.markdown(">Probabilité du repaiement:")
            st.write(API_data['probability1'])
            st.markdown('>Seuil maximal:')
            st.markdown(threshold)
            st.markdown(">Décision sur la demande de crédit:")
            st.markdown(etat_1)
            decision = 1

# Interprétabilité du résultat avec SHAP explainer

    st.write('## Influence des variables')
    shap.initjs()
    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st_shap(shap.force_plot(exp_val[1], shap_val[1][idx,:], X.iloc[idx,:]))

# Comparaison des données des clients avec des clients similaires

    st.write('## Comparaison avec des clients similaires')
    comparaison_option = st.selectbox('Comparaison des clients', ('Source Externe 2', 'Enquetes de crédit',
                                                                  'Informations de contact laissé par le client',
                                                                  "Situation de famille:Marié(e) ou pas",
                                                                  "Employé en tant qu'ouvrier"))
    data_client = data_client[data_client['SK_ID_CURR'] == identifiant]
    if decision == 1:
        st.write('>Comparaison avec des clients dont la demande de crédit a été accordé')
        if comparaison_option == "Source Externe 2":
            col = 'EXT_SOURCE_2'
            client_val = data_client[col].values[0]
            st.write("La valeur 'EXT_SOURCE_2' du client est: ", client_val)
            plot(accord, col, client_val)

        if comparaison_option == "Enquetes de crédit":
            col = 'Enquiry'
            client_val = data_client[col].values[0]
            st.write("La valeur 'Enquiry' du client est: ", client_val)
            plot(accord, col, client_val)

        if comparaison_option == "Informations de contact laissé par le client":
            col = 'Contact'
            client_val = data_client[col].values[0]
            st.write("La valeur 'Contact' du client est: ", client_val)
            plot(accord, col, client_val)

        if comparaison_option == "Situation de famille:Marié(e) ou pas":
            col = 'NAME_FAMILY_STATUS_Married'
            client_val = data_client[col].values[0]
            st.write("La valeur 'NAME_FAMILY_STATUS_Married' du client est: ", client_val)
            plot(accord, col, client_val)

        if comparaison_option == "Employé en tant qu'ouvrier":
            col = 'OCCUPATION_TYPE_Laborers'
            client_val = data_client[col].values[0]
            st.write("La valeur 'OCCUPATION_TYPE_Laborers' du client est: ", client_val)
            plot(accord, col, client_val)
    else:
        st.write('>Comparaison avec des clients dont la demande de crédit a été refusé')
        if comparaison_option == "Source Externe 2":
            col = 'EXT_SOURCE_2'
            client_val = data_client[col].values[0]
            st.write("La valeur 'EXT_SOURCE_2' du client est: ", client_val)
            plot(refus, col, client_val)

        if comparaison_option == "Enquetes de crédit":
            col = 'Enquiry'
            client_val = data_client[col].values[0]
            st.write("La valeur 'Enquiry' du client est: ", client_val)
            plot(refus, col, client_val)

        if comparaison_option == "Informations de contact laissé par le client":
            col = 'Contact'
            client_val = data_client[col].values[0]
            st.write("La valeur 'Contact' du client est: ", client_val)
            plot(refus, col, client_val)

        if comparaison_option == "Situation de famille:Marié(e) ou pas":
            col = 'NAME_FAMILY_STATUS_Married'
            client_val = data_client[col].values[0]
            st.write("La valeur 'NAME_FAMILY_STATUS_Married' du client est: ", client_val)
            plot(refus, col, client_val)

        if comparaison_option == "Employé en tant qu'ouvrier":
            col = 'OCCUPATION_TYPE_Laborers'
            client_val = data_client[col].values[0]
            st.write("La valeur 'OCCUPATION_TYPE_Laborers' du client est: ", client_val)
            plot(refus, col, client_val)