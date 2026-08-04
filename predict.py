import joblib

model = joblib.load(
    "models/diabetes_model.pkl"
)

def predict(features):

    return model.predict(features)