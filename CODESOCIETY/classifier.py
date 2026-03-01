import pickle
import os

class FakeNewsClassifier:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(base_path, "model.pkl")
        vectorizer_path = os.path.join(base_path, "vectorizer.pkl")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict_proba(self, text):
        vector = self.vectorizer.transform([text])
        return self.model.predict_proba(vector)