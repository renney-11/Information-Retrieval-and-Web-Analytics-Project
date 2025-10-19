import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if it is not already downloaded
nltk.download('stopwords')

# Initialize stemmer and stopwords
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def preprocess_text(text: str) -> str:
    """
    Preprocesseses a text field for indexing and retrieval by:
    1. Converting to loweercase
    2. Removing punctuation
    3. Tokenizing
    4. Removing stopwords
    5. Applying stemming
    :param text:
    :return:
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('','', string.punctuation))
    # Tokenize, remove stopwords, and apply stemming
    tokens = [STEMMER.stem(word) for word in text.split() if word not in STOP_WORDS]
    # Join tokens back to a single string
    return " ".join(tokens)


def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = pd.read_json(path)
    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe, preprocess title and description fields
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        # Preprocess title and description
        row['title'] = preprocess_text(row.get('title', ''))
        row['description'] = preprocess_text(row.get('description', ''))

        # Create Doc object with all fields
        doc = Document(
            pid=row.get('pid'),
            title=row.get('title'),
            description=row.get('description'),
            brand=row.get('brand'),
            category=row.get('category'),
            sub_category=row.get('sub_category'),
            product_details=row.get('product_details'),
            seller=row.get('seller'),
            out_of_stock=row.get('out_of_stock'),
            selling_price=row.get('selling_price'),
            discount=row.get('discount'),
            actual_price=row.get('actual_price'),
            average_rating=row.get('average_rating'),
            url=row.get('url')
        )
        corpus[doc.pid] = doc
    return corpus

