# Importation des libraries

import uvicorn
from fastapi import FastAPI
import pandas as pd
import pickle
import re

# FastAPI

app = FastAPI(title="Prêt à dépenser",
              description="Afficher la probabilité qu'un client échoue de rembourser un prêt",
              version="0.9.0")

# Importation du dataframe

app_data = pd.read_csv('app_tr.csv')
app_data = app_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Préparation du dataframe

X = app_data.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
y = app_data['TARGET']

# Importation du modèle

model = pickle.load(open('Pickle_LGBM_Model.pkl', 'rb'))
model.fit(X, y)

# Path pour la fonction main

@app.get('/')
def main():
    return {'message': 'Bienvenue à Prêt à dépenser!'}

# Vérification de la validité de l'ID du client

@app.get("/api/clients/{id}")
async def client_details(id: int):
    clients_id = app_data["SK_ID_CURR"].tolist()
    if id not in clients_id:
        return "Client ID non valide"
    else:
        return "Client ID valide"

# Décision sur la demande de prêt d'argent à l'aide du modèle

@app.get("/api/clients/{id}/display")
async def display_client(id: int):
    clients_id = app_data["SK_ID_CURR"].tolist()
    dis_client = app_data[app_data.SK_ID_CURR.values == id]
    dis_client = dis_client.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    threshold = 0.14

# Vérification de l'ID du client

    if id not in clients_id:
        return "client's id not found"

# Prédiction avec le modèle

    else:
        result_proba = model.predict_proba(dis_client)
        y_prob = result_proba[:, 1]
        result = (y_prob >= threshold).astype(int)
        if (int(result[0]) == 0):
            result = "Client peu risqué: Demande de crédit accordé"
        else:
            result = "Client risqué: Demande de crédit refusé"
    return {'result': result,
            "probability0" : result_proba[0][0],
            "probability1" : result_proba[0][1],
            "threshold" : threshold }

# Adresse IP pour l'exécution de l'application

uvicorn.run(app, host="127.0.0.1", port=8000)