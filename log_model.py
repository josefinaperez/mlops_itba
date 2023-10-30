# %%
import mlflow
from model import NER_HFModel

# %%
mlflow.set_experiment('ner_hf')
mlflow.start_run()
# %%
model_checkpoint = "dslim/bert-base-NER"
mlflow.log_param('model_checkpoint', model_checkpoint)
# %%
model = NER_HFModel(model_checkpoint)
# %%
# %%
# %%
mlflow.pyfunc.log_model(
    artifact_path='model',
    python_model=model,
    registered_model_name='NER_HF',
    code_path=['model_src'],
    conda_env='conda.yaml',
)
# %%
mlflow.end_run()
# %% Test model

model.load_context(None)
# %%
import pandas as pd
df = pd.DataFrame([
    {'text': 'Hola que tal, mi nombre es "Julian".'}
    ])
df
# %%
from model_src.preprocess import preprocess_text
# %%
preprocess_text(df)
# %%
df
# %%
# %%

model.predict(None, df)