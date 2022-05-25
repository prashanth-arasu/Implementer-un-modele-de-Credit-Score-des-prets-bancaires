import uvicorn
from fastapi import FastAPI
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import re

app = FastAPI(title="Prêt à dépenser",
              description="Afficher la probabilité qu'un client échoue de rembourser un prêt",
              version="0.9.0")

app_data = pd.read_csv('/app/FastAPI/app_tr.csv')

app_data = app_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


X = app_data.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
y = app_data['TARGET']

model = pickle.load(open('/app/FastAPI/Pickle_LGBM_Model.pkl', 'rb'))
#model = LGBMClassifier(colsample_bytree=0.5821950337934017, min_child_samples=279,
#               min_child_weight=10000, num_leaves=41, reg_alpha=0,
#               reg_lambda=0.5, subsample=0.9440571485798224)
model.fit(X, y)

# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Bienvenue à Prêt à dépenser!'}

# Defining path operation for root endpoint


@app.get("/api/clients/{id}")
async def client_details(id: int):

    clients_id = app_data["SK_ID_CURR"].tolist()

    if id not in clients_id:
        return "Client ID non valide"
    else:
        return "Client ID valide"

@app.get("/api/clients/{id}/display")
async def display_client(id: int):
    clients_id = app_data["SK_ID_CURR"].tolist()
    dis_client = app_data[app_data.SK_ID_CURR.values == id]
    dis_client = dis_client.drop(['SK_ID_CURR', 'TARGET'], axis = 1)
    threshold = 0.14
    if id not in clients_id:
        return "client's id not found"
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

uvicorn.run(app, host="127.0.0.1", port=8000)