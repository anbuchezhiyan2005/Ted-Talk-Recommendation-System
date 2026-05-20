import string
import warnings
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")

# Ensure stopwords are available (quietly download if missing)
nltk.download("stopwords", quiet=True)


# --- Load and preprocess data once at startup ---

RAW_CSV_PATH = "Data/tedx_datase.csv"


def _load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV_PATH)

    # Combine title and details as in the notebook for richer text
    df["combined_text"] = df["title"].fillna("") + " " + df["details"].fillna("")

    # Keep useful columns for display
    df = df[["main_speaker", "title", "details", "url", "combined_text"]].dropna(
        subset=["combined_text"]
    )
    return df


_raw_df = _load_raw_data()


def _remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words("english"))
    important_words: List[str] = []
    for word in str(text).split():
        word = word.lower()
        if word and word not in stop_words:
            important_words.append(word)
    return " ".join(important_words)


_PUNCTUATIONS = string.punctuation


def _clean_punctuations(text: str) -> str:
    table = str.maketrans("", "", _PUNCTUATIONS)
    return str(text).translate(table)


# Apply the same style of preprocessing as in the notebook
_processed_df = _raw_df.copy()
_processed_df["processed_text"] = _processed_df["combined_text"].apply(_remove_stopwords)
_processed_df["processed_text"] = _processed_df["processed_text"].apply(_clean_punctuations)


# Fit TF-IDF on processed text
_vectorizer = TfidfVectorizer(analyzer="word")
_vectorizer.fit(_processed_df["processed_text"])


def _get_similarities(processed_query: str):
    """
    Compute cosine similarity and Pearson correlation between the query
    and all talks using the fitted TF-IDF vectors.
    """
    query_vec = _vectorizer.transform([processed_query]).toarray()

    sims = []
    pears = []

    for _, row in _processed_df.iterrows():
        talk_vec = _vectorizer.transform([row["processed_text"]]).toarray()

        cos_sim = cosine_similarity(query_vec, talk_vec)[0][0]
        pea_sim = pearsonr(query_vec.squeeze(), talk_vec.squeeze())[0]

        sims.append(cos_sim)
        pears.append(pea_sim)

    return sims, pears


def recommend_talks(query: str, top_k: int = 5) -> pd.DataFrame:
    """
    Recommend top_k TED talks for the given free-text query.

    Returns a DataFrame with:
      - title
      - main_speaker
      - short_description (shortened version of full details)
      - url
    """
    if not query or not query.strip():
        return _processed_df.head(0)[
            ["title", "main_speaker", "details", "url"]
        ]  # empty

    processed_query = _clean_punctuations(_remove_stopwords(query))

    sims, pears = _get_similarities(processed_query)

    scored_df = _processed_df.copy()
    scored_df["cos_sim"] = sims
    scored_df["pea_sim"] = pears

    scored_df.sort_values(
        by=["cos_sim", "pea_sim"], ascending=[False, False], inplace=True
    )

    top = scored_df.head(top_k).copy()

    # Build a short description
    def _shorten(text: str, max_len: int = 220) -> str:
        text = str(text)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rsplit(" ", 1)[0] + "..."

    top["short_description"] = top["details"].apply(_shorten)

    return top[["title", "main_speaker", "short_description", "url"]]


__all__ = ["recommend_talks"]


