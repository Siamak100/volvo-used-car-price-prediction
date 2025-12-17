import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# 1. Load data
# -----------------------------
df = pd.read_excel("data_insamling_volvo_blocket.xlsx")

# -----------------------------
# 2. Handle missing values
# -----------------------------
# Separate numeric and categorical columns
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

# Fill missing numeric values with median
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill missing categorical values with placeholder
df[cat_cols] = df[cat_cols].fillna("Unknown")

# -----------------------------
# 3. Define target and features
# -----------------------------
y = df["Försäljningspris"]
X = df.drop(columns=["Försäljningspris"])

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# 4. Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Define models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=300, max_iter=10000),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
}

# -----------------------------
# -----------------------------
# 6. Train models and evaluate
# -----------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "R2": r2
    })

    #print(f"{name}")
    #print(f"  RMSE: {rmse:.2f}")
    #print(f"  R2:   {r2:.3f}")
    #print("-" * 30)


# -----------------------------
# 7. Results summary table
# -----------------------------
results_df = pd.DataFrame(results)
print("\nModel comparison:")
print(results_df)


