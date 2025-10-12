import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")
iris = load_iris()
_, X_test, _, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

print("Testing - > ")

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"✅ Model accuracy: {acc:.2f}")
if acc < 0.9:
    raise Exception("❌ Accuracy below acceptable threshold!")