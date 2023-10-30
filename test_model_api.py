# %%
# mlflow models serve -m models:/NER_HF/4 --host 0.0.0.0 --port 5000 --env-manager=local
# %%
import requests
# %%
import pandas as pd
df = pd.DataFrame([
    {'text': 'Hola que tal, soy Julian.'}
])
# %%
json_data = {'dataframe_records': df.to_dict(orient='records')}
json_data
# %%
url = 'http://localhost:5000/invocations'

response = requests.post(
    url,
    json=json_data
)
# %%
response.json()
# %% en docker
# mlflow models build-docker -m models:/NER_HF/4 -n ner-docker --env-manager conda --install-mlflow