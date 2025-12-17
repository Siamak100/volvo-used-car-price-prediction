import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def compare_models(X, y, test_size=0.2, random_state=42):
    """
    Train and compare multiple regression models using RMSE and R2.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = mean_squared_error(y_test, predictions, squared=False)
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "RMSE": rmse,
            "R2": r2
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage (requires prepared X, y)
    print("This module is intended to be imported and used from modeling.py")
