from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import spacy

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, nlp=None):
        # Use provided spaCy model or load the default English model
        self.nlp = nlp or spacy.load("en_core_web_sm")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a pandas Series for consistent processing
        X = pd.Series(X)
        return X.apply(self.clean_text)

    def clean_text(self, text):
        # Lowercase, lemmatize, and keep only alphabetic tokens
        doc = self.nlp(text.lower())
        clean_tokens = [token.lemma_ for token in doc if token.is_alpha]
        return " ".join(clean_tokens)