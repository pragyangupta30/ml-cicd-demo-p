# testmodel.py

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")

data = load_breast_cancer()
_, X_test, _, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model accuracy: {acc:.2f}")
if acc < 0.9:
    raise Exception("❌ Accuracy below acceptable threshold!")
