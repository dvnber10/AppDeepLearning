import numpy as np
import pandas as pd

train = pd.read_csv("titanic/train.csv")
test = pd.read_csv("titanic/test.csv")

# procesamiento
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
train["Age"] = train.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
test["Age"] = test.groupby(["Pclass", "Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
test["Fare"].fillna(test["Fare"].median(), inplace=True)
train["Ticket"] = train["Cabin"].notnull().astype(int)
test["Ticket"] = test["Cabin"].notnull().astype(int)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
test["IsAlone"] = (test["FamilySize"] == 1).astype(int)
train["HasCabin"] = train["Cabin"].notna().astype(int)
test["HasCabin"] = test["Cabin"].notna().astype(int)
embarked_map = {"S": 0, "C": 1, "Q": 2}
train["Embarked"] = train["Embarked"].map(embarked_map)
test["Embarked"] = test["Embarked"].map(embarked_map)
train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# Normalizar un poco los títulos más raros
for df in [train, test]:
    df["Title"] = df["Title"].replace(
        ["Mlle", "Ms"], "Miss"
    ).replace("Mme", "Mrs")

# Codificar como números
title_map = {title: idx for idx, title in enumerate(train["Title"].unique())}
train["Title"] = train["Title"].map(title_map)
test["Title"] = test["Title"].map(title_map)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Ticket", "FamilySize", "IsAlone", "HasCabin"]

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
epochs = 50000

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
submission.to_csv("submission4.csv", index=False)
