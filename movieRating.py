import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("IMDb Movies India.csv", encoding="latin-1")

df.dropna(subset=["Rating"], inplace=True)

df["Duration"] = df["Duration"].astype(str).str.replace(" min", "")
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")

df["Votes"] = df["Votes"].astype(str).str.replace(",", "")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

required_columns = ["Genre", "Director", "Actor 1", "Actor 2", "Duration", "Votes", "Rating"]
df = df[required_columns].dropna()

x = df.drop("Rating", axis=1)
y = df["Rating"]

categorical_features = ["Genre", "Director", "Actor 1", "Actor 2"]
numerical_features = ["Duration", "Votes"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model.fit(x_train, y_train)

predictions = model.predict(x_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}\n")

new_movie = pd.DataFrame({
    "Genre": ["History"],
    "Director": ["Laxman Utekar"],
    "Actor 1": ["Vicky Kaushal"],
    "Actor 2": ["Rashmika Mandana"],
    "Duration": [161],
    "Votes": [45000],  
})

predicted_rating = model.predict(new_movie)
print(f"Predicted Rating for the new movie: {predicted_rating[0]:.2f}\n")

plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], bins=20, kde=True)
plt.title("Distribution of Movie Ratings")
plt.show()

plt.figure(figsize=(10, 5))
df["Genre"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Movie Genres")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x="Votes", y="Rating", data=df, alpha=0.5)
plt.title("Votes vs. Rating")
plt.xscale('log')
plt.show()