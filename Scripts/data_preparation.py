import os, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from huggingface_hub import HfApi, login
from datasets import load_dataset

login(token=os.environ["HF_TOKEN"])
api  = HfApi()
REPO = os.environ.get("HF_DATASET_REPO","naidu1999/tourism-data")

df = load_dataset(REPO, data_files="raw_data.csv", split="train").to_pandas()
if "CustomerID" in df.columns: df.drop(columns=["CustomerID"], inplace=True)
df.drop_duplicates(inplace=True)
for c in df.select_dtypes(include=["float64","int64"]).columns:
    if c != "ProdTaken": df[c].fillna(df[c].median(), inplace=True)
for c in df.select_dtypes(include="object").columns:
    df[c].fillna(df[c].mode()[0], inplace=True)
enc_map = {}
for c in df.select_dtypes(include="object").columns:
    le = LabelEncoder(); df[c] = le.fit_transform(df[c].astype(str))
    enc_map[c] = {str(cls):int(i) for i,cls in enumerate(le.classes_)}
json.dump(enc_map, open("encoding_map.json","w"), indent=2)
X,y = df.drop(columns=["ProdTaken"]), df["ProdTaken"]
X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
X_tr,y_tr = SMOTE(random_state=42).fit_resample(X_tr,y_tr)
pd.concat([pd.DataFrame(X_tr,columns=X.columns),pd.Series(y_tr,name="ProdTaken")],axis=1).to_csv("train.csv",index=False)
pd.concat([X_te.reset_index(drop=True),y_te.reset_index(drop=True)],axis=1).to_csv("test.csv",index=False)
for f in ["train.csv","test.csv","encoding_map.json"]:
    api.upload_file(path_or_fileobj=f,path_in_repo=f,repo_id=REPO,repo_type="dataset")
print("Data prep done.")
