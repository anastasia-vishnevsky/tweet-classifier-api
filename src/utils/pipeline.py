import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.text_cleaner import TextCleaner

def create_pipeline():
    nlp = spacy.load("en_core_web_sm")
    return Pipeline([
        ('cleaner', TextCleaner(nlp)),
        ('tfidf', TfidfVectorizer())
    ])


