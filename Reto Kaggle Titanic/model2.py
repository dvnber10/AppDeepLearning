import numpy as np
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")

# procesamiento
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

X = train[features].values
y = train["Survived"].values.reshape(-1, 1)

X = (X - X.mean(axis=0)) / X.std(axis=0)

# Inicializar los pesos
m, n = X.shape
w = np.zeros((n, 1))
b = 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

alpha = 0.01
epochs = 10000

for i in range(epochs):
    z = np.dot(X, w) + b
    y_hat = sigmoid(z)

    cost = - (1/m) * np.sum(y * np.log(y_hat + 1e-9) + (1-y) * np.log(1 - y_hat + 1e-9))

    dw = (1/m) * np.dot(X.T, (y_hat - y))
    db = (1/m) * np.sum(y_hat - y)

    w -= alpha * dw
    b -= alpha * db

    if i % 100 == 0:
        print(f"Epoch {i}, Costo: {cost:.4f}")

X_test = test[features].values
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

preds = sigmoid(np.dot(X_test, w) + b)
preds = (preds > 0.5).astype(int)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": preds.flatten()
})
submission.to_csv("submission2.csv", index=False)
