import os
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
api        = HfApi()
SPACE_REPO = os.environ.get("HF_SPACE_REPO","naidu1999/tourism-app")

api.create_repo(repo_id=SPACE_REPO,repo_type="space",space_sdk="docker",exist_ok=True,private=False)
for f in ["app.py","requirements.txt","Dockerfile"]:
    api.upload_file(path_or_fileobj=os.path.join("deployment",f),
                    path_in_repo=f,repo_id=SPACE_REPO,repo_type="space")
    print(f"Uploaded {f}")
print(f"Deployed: https://huggingface.co/spaces/{SPACE_REPO}")
