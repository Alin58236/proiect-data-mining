from flask import Flask, request #type: ignore[import]
import pandas as pd
from flask_cors import CORS #type: ignore[import]
from Models import trainModel
from common import preprocess

app = Flask(__name__)
CORS(app, supports_credentials=True)

def loadData():
    dfs = []
    for i in range(1,5):
        path = './data/UNSW-NB_complet/UNSW-NB15_{}.csv'  
        dfs.append(pd.read_csv(path.format(i), header=None, low_memory=False))
    df = pd.concat(dfs).reset_index(drop=True)

    df_col = pd.read_csv('./data/UNSW-NB_complet/NUSW-NB15_features.csv', encoding='ISO-8859-1')
    df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())
    df.columns = df_col['Name']

    cols_to_drop_total = [
        'srcip', 'dstip', 'stime', 'ltime',
        'dsport', 'sport',
        'attack_cat',
        'stcpb', 'dtcpb',
        'trans_depth', 'res_bdy_len',
        'is_ftp_login', 'ct_flw_http_mthd', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm'
    ]
    df.drop(columns=cols_to_drop_total, inplace=True)

    df = df.drop_duplicates().reset_index(drop=True)

    print("Data loaded successfully.")
    return df


df = None
if(df is None):
    df = loadData()

@app.route("/", methods=["GET"])
def status_check():
    return {
        "status": "healthy",
        "message": "ML API server is running",
        "endpoint": "/train_model (POST)"
    }, 200

@app.route("/train_model", methods=["POST"])
def train_model():
    global df
    data = request.get_json()

    print(f"Parsed JSON:\n {data}")
    print("========================")

    model_name = data.get("model_name", "")
    if model_name == "":
        return {"error": "Model name is required."}, 400
    if model_name == "DecisionTree":
        params = None
    else:
        params = data.get("params", {})

    print(f"Training model: {model_name} with params: {params}")

    X_scaled, y, _, _ = preprocess(df)
    print("Preprocess completed, training model...")
    result = trainModel(model_name, X_scaled, y, params)
    print("Training completed!")
    
    return result, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)