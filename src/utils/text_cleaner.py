from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import spacy

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X)
        return X.apply(self.clean_text)

    def clean_text(self, text):
        doc = self.nlp(text.lower())
        clean_tokens = [token.lemma_ for token in doc if token.is_alpha]
        return " ".join(clean_tokens)