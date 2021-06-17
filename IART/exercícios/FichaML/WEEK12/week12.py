import pandas as pd
from pathlib import Path
import os
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

base_path = Path(__file__).parent

def example_weather_nominal():
    path = (base_path / "weather-nominal.csv").resolve();
    series = pd.read_csv(path)
    # arrange table in X(features) and y(target)
    X = series.iloc[:, :-1]
    X = X.apply(LabelEncoder().fit_transform)
    y = series.iloc[:, -1]
    # apply GaussianNB and CategoricalNB
    gNB = GaussianNB()
    gNB.fit(X, y)
    cNB = CategoricalNB()
    cNB.fit(X, y)
    print(f"Prediction GaussianNB ([Sunny,Cool,High,True]]): {gNB.predict([[2,0,0,1]])}")
    print(f"Probability GaussianNB: {gNB.predict_proba([[2,0,0,1]])}")
    print("\n")
    print(f"Prediction CategoricalNB ([Sunny,Cool,High,True]]): {cNB.predict([[2, 0, 0, 1]])}")
    print(f"Probability CategoricalNB: {cNB.predict_proba([[2, 0, 0, 1]])}")


def example_weather_numeric():
    path = (base_path / "weather-numeric.csv").resolve();
    series = pd.read_csv(path)
    # arrange table in X(features) and y(target)
    X = series.iloc[:, :-1]
    X.outlook = LabelEncoder().fit_transform(X.outlook)
    X.windy = LabelEncoder().fit_transform(X.windy)
    y = series.iloc[:, -1]
    # apply GaussianNB and CategoricalNB
    gNB = GaussianNB()
    gNB.fit(X, y)
    cNB = CategoricalNB()
    cNB.fit(X, y)
    print(f"Prediction GaussianNB ([Sunny,66,90,True]]]): {gNB.predict([[2, 66, 90, 1]])}")
    print(f"Probability GaussianNB: {gNB.predict_proba([[2, 66, 90, 1]])}")
    print("\n")
    print(f"Prediction CategoricalNB ([Sunny,66,90,True]]): {cNB.predict([[2, 66, 90, 1]])}")
    print(f"Probability CategoricalNB: {cNB.predict_proba([[2, 66, 90, 1]])}")


if __name__ == "__main__":
    example_weather_nominal()
    print("\n-------------------\n")
    example_weather_numeric()