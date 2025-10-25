import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv('titanic/train.csv')
df_test = pd.read_csv('titanic/test.csv')

df_train = pd.get_dummies(df_train, columns=["Sex", "Embarked"])
df_test = pd.get_dummies(df_test, columns=["Sex", "Embarked"])

df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

features = [col for col in df_train.columns if col not in ["PassengerId", "Survived", "Name", "Ticket", "Cabin"]]

x = df_train[features]
y = df_train["Survived"]

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_val, y_val))

# Generar predicciones

predictions = model.predict(df_test[features])

submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)