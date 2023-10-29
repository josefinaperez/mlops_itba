# %%
import pandas as pd
import numpy as np
import mlflow
from matplotlib import pyplot as plt
# %%
mlflow
mlflow.set_experiment("EDA")
# %%
prefix = "files/data/"
scores_url = prefix + "scores_0.csv"
peliculas_url = prefix + "peliculas_0.csv"
personas_url = prefix + "personas_0.csv"
trabajadores_url = prefix + "trabajadores_0.csv"
usuarios_url = prefix + "usuarios_0.csv"

df_scores = pd.read_csv(scores_url)
df_personas = pd.read_csv(personas_url)
df_usuarios = pd.read_csv(usuarios_url)
df_trabajadores = pd.read_csv(trabajadores_url)
df_peliculas = pd.read_csv(peliculas_url)
df_personas["year of birth"] = df_personas["year of birth"].apply(lambda x:int(x))
# %%
#mlflow.log_artifacts("data")
# %%
#Grafico el histograma de scores
fig = plt.figure(figsize=(9,3))
plt.hist(df_scores.rating, bins=5)
plt.title(f"Histograma de los ratings. Promedio = {df_scores.rating.mean()}")
mlflow.log_figure(fig, "Histograma ratings.png")

# %%
# Voy a buscar correlación entre fecha de nacimiento/género y score
df_merge = df_scores.merge(df_personas,left_on="user_id", right_on="id")
df_mean = df_merge.groupby(["year of birth", "Gender"]).rating.mean().reset_index()
fig = plt.figure(figsize=(9,3))
df_mean.query("Gender == 'M'").set_index("year of birth").rating.plot(label="Male")
df_mean.query("Gender == 'F'").set_index("year of birth").rating.plot(label="Female")
plt.legend()
plt.ylabel("Rating")
mlflow.log_figure(fig, "Rating promedio por año.png")
# %%
# Guardo como métrica el score promedio
mlflow.log_metric("Avg Score", df_scores.rating.mean(),)
mlflow.log_metric("Min Score", df_scores.rating.min())
mlflow.log_metric("Max Score", df_scores.rating.max())
mlflow.log_metric("Score Std", df_scores.rating.std())
mlflow.log_metric("Sparcity", len(df_scores)/(len(df_personas)*len(df_peliculas)))
# %%



mlflow.end_run()

# %%