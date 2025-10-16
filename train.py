# # train.py
# import joblib
# from sklearn.datasets import load_breast_cancer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# data = load_breast_cancer()
# X_train, _, y_train, _ = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# joblib.dump(model, "model.pkl")
# print("✅ Model trained and saved as model.pkl")
