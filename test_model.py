# %%
import mlflow
logged_model = 'models:/NER_HF/2'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# %%
# %%
# Predict on a Pandas DataFrame.
import pandas as pd
df = pd.DataFrame([
    {'text': 'Hola que tal, soy Julian.'}
])
# %%
loaded_model.predict(df)
# %%