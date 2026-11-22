"""
sentiment.py  -  labels, counts and word clouds for the cleaned 3-column data
(post_id | title | post) produced by clean_data.load_and_clean().

Colab quick start:
    !pip install vaderSentiment wordcloud
    from clean_data import load_and_clean
    from sentiment import analyze, sentiment_counts, count_sentence, plot_clouds

    df = load_and_clean("your_file.csv")
    df = analyze(df)                 # adds: text, score, sentiment
    print(count_sentence(df))
    plot_clouds(df)
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, WordCloud

LABELS = ["Positive", "Negative", "Neutral"]
_analyzer = SentimentIntensityAnalyzer()


# ------------------------------------------------------------------ 1. sentiment
def build_text(df: pd.DataFrame) -> pd.Series:
    """Title first, then the post. Works even if the title or post is empty."""
    return (df["title"].fillna("") + ". " + df["post"].fillna("")).str.strip(" .")


def score_text(text: str) -> float:
    """VADER compound score in [-1, +1]. Empty text scores 0."""
    return _analyzer.polarity_scores(text)["compound"] if text else 0.0


def to_label(score: float, neutral_zone: float = 0.05) -> str:
    if score >= neutral_zone:
        return "Positive"
    if score <= -neutral_zone:
        return "Negative"
    return "Neutral"


def analyze(df: pd.DataFrame, neutral_zone: float = 0.05) -> pd.DataFrame:
    """Adds columns: text, score, sentiment. Returns a new DataFrame."""
    out = df.copy()
    out["text"] = build_text(out)
    out["score"] = out["text"].map(score_text)
    out["sentiment"] = out["score"].map(lambda s: to_label(s, neutral_zone))
    return out


# ------------------------------------------------------------------ 2. counts
def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    """Counts in fixed order Positive, Negative, Neutral (zeros included)."""
    return df["sentiment"].value_counts().reindex(LABELS, fill_value=0)


def count_sentence(df: pd.DataFrame) -> str:
    c, n = sentiment_counts(df), len(df)
    if n == 0:
        return "No posts to analyse."
    return (f"Out of {n:,} posts analysed, {c['Positive']:,} were positive "
            f"({c['Positive'] / n:.1%}), {c['Negative']:,} were negative "
            f"({c['Negative'] / n:.1%}) and {c['Neutral']:,} were neutral "
            f"({c['Neutral'] / n:.1%}).")


def plot_pie(df: pd.DataFrame):
    c = sentiment_counts(df)
    colors = {"Positive": "#2e9e5b", "Negative": "#d64545", "Neutral": "#8a94a6"}
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(c.values, labels=c.index, colors=[colors[k] for k in c.index],
           autopct=lambda p: f"{p:.1f}%" if p > 0 else "", startangle=90,
           wedgeprops={"width": 0.45, "edgecolor": "white"})
    ax.set_title("Sentiment share")
    return fig


# ------------------------------------------------------------------ 3. word clouds
def auto_stopwords(texts, max_doc_freq: float = 0.4, min_docs: int = 30) -> set:
    """
    Words that appear in more than `max_doc_freq` of all posts carry no information
    (e.g. 'uoft' in a UofT dataset). Switched off for tiny datasets.
    """
    texts = list(texts)
    if len(texts) < min_docs:
        return set()
    docs = [set(re.findall(r"[a-z']{3,}", t.lower())) for t in texts]
    counts = Counter(w for d in docs for w in d)
    return {w for w, c in counts.items() if c / len(docs) > max_doc_freq}


def get_stopwords(df: pd.DataFrame, max_doc_freq: float = 0.4, extra=()) -> set:
    """Built-in English list + auto-detected common words + anything the user adds."""
    texts = df["text"] if "text" in df else build_text(df)
    return set(STOPWORDS) | auto_stopwords(texts, max_doc_freq) | {w.lower() for w in extra}


def make_cloud(texts, title: str, stop: set):
    """Returns a matplotlib figure, or None if there isn't enough text."""
    text = " ".join(texts).lower()
    if len(text.split()) < 5:
        return None
    wc = WordCloud(width=900, height=450, background_color="white", stopwords=stop,
                   max_words=100, collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return fig


def plot_clouds(df: pd.DataFrame, max_doc_freq: float = 0.4, extra_stop=(), show=True):
    """
    One cloud for all posts + one per sentiment class. Stopwords are computed once on
    the full dataset so the clouds stay comparable. Returns {name: figure}.
    """
    stop = get_stopwords(df, max_doc_freq, extra_stop)
    figs = {"All posts": make_cloud(df["text"], "All posts", stop)}
    for label in LABELS:
        figs[label] = make_cloud(df.loc[df["sentiment"] == label, "text"],
                                 f"{label} posts", stop)
    if show:
        for name, fig in figs.items():
            if fig is None:
                print(f"Not enough text for '{name}'")
        plt.show()
    return figs


def top_words(df: pd.DataFrame, label: str, n: int = 15, max_doc_freq: float = 0.4) -> list:
    """Most frequent words in one sentiment class (after stopword removal)."""
    stop = get_stopwords(df, max_doc_freq)
    words = re.findall(r"[a-z']{3,}", " ".join(df.loc[df["sentiment"] == label, "text"]).lower())
    return Counter(w for w in words if w not in stop).most_common(n)
