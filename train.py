import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_model_training():
    # 1. Load the Iris Dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 2. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train a basic machine learning model
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, y_train)

    # 4. Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model Accuracy: {score * 100:.2f}%")

    # 5. Continuous Integration (CI) Check: Ensure accuracy is acceptable
    assert score > 0.80, f"Model accuracy is too low: {score}"

if __name__ == "__main__":
    test_model_training()