import os, json, joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBClassifier
from huggingface_hub import HfApi, login
from datasets import load_dataset

login(token=os.environ["HF_TOKEN"])
api      = HfApi()
DATA_REPO  = os.environ.get("HF_DATASET_REPO","naidu1999/tourism-data")
MODEL_REPO = os.environ.get("HF_MODEL_REPO","naidu1999/tourism-model")

tr = load_dataset(DATA_REPO,data_files="train.csv",split="train").to_pandas()
te = load_dataset(DATA_REPO,data_files="test.csv",split="train").to_pandas()
X_tr,y_tr = tr.drop(columns=["ProdTaken"]),tr["ProdTaken"]
X_te,y_te = te.drop(columns=["ProdTaken"]),te["ProdTaken"]

gs = GridSearchCV(XGBClassifier(random_state=42,eval_metric="logloss",verbosity=0),
    {"n_estimators":[100,200],"max_depth":[3,5],"learning_rate":[0.05,0.1]},
    cv=5,scoring="roc_auc",n_jobs=-1)
gs.fit(X_tr,y_tr)
model = gs.best_estimator_
auc = roc_auc_score(y_te,model.predict_proba(X_te)[:,1])
acc = accuracy_score(y_te,model.predict(X_te))
joblib.dump(model,"model.pkl")
json.dump(X_tr.columns.tolist(),open("feature_names.json","w"))
json.dump({"model_name":"XGBoost","test_auc":auc,"test_accuracy":acc,
           "best_params":gs.best_params_,"features":X_tr.columns.tolist()},
          open("metadata.json","w"),indent=2)
api.create_repo(repo_id=MODEL_REPO,repo_type="model",exist_ok=True,private=False)
for f in ["model.pkl","feature_names.json","metadata.json"]:
    api.upload_file(path_or_fileobj=f,path_in_repo=f,repo_id=MODEL_REPO,repo_type="model")
print(f"Model done. AUC={auc:.4f} Acc={acc:.4f}")
