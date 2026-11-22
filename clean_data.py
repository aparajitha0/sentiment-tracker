"""
clean_data.py  -  load and clean ANY posts dataset down to 3 columns:

    post_id | title | post

* post_id : the column that uniquely identifies each row; serial numbers (1..N) if none exists
* title   : the title column if the dataset has one, else empty strings
* post    : the main post text (cleaned)

Usage (Colab or script):
    from clean_data import load_and_clean
    df = load_and_clean("your_file.csv")
"""
import html
import re

import pandas as pd

# ------------------------------------------------------------------ name hints
ID_HINTS = ["post_id", "id", "tweet_id", "review_id", "comment_id", "uid", "row_id", "index"]
TITLE_HINTS = ["title", "headline", "subject", "post_title"]
POST_HINTS = ["content", "text", "post", "body", "selftext", "post_text", "review",
              "comment", "tweet", "message", "description", "full_text"]
PLACEHOLDERS = {"", "nan", "none", "null", "n/a", "na", "[deleted]", "[removed]"}


# ------------------------------------------------------------------ loading
def load_dataset(path: str) -> pd.DataFrame:
    """Read csv / tsv / xlsx / json, trying several encodings for text files."""
    p = path.lower()
    if p.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if p.endswith(".json"):
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    sep = "\t" if p.endswith((".tsv", ".tab")) else ","
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, sep=sep)
            if df.shape[1] == 1 and sep == ",":      # maybe ; or | separated - let pandas sniff
                df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:                # malformed lines: skip them instead of failing
            return pd.read_csv(path, encoding=enc, sep=sep, engine="python",
                               on_bad_lines="skip")
    raise ValueError(f"Could not read {path}")


# ------------------------------------------------------------------ column detection
def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find(df, hints, exclude=()):
    """First column whose normalised name matches a hint (in hint priority order)."""
    names = {_norm(c): c for c in df.columns if c not in exclude}
    for h in hints:
        if h in names:
            return names[h]
    return None


def detect_id_col(df):
    """A column is a valid id only if every value is present and unique."""
    candidates = []
    hit = _find(df, ID_HINTS)
    if hit is not None:
        candidates.append(hit)
    candidates += [c for c in df.columns if c not in candidates
                   and re.search(r"(^|_)id($|_)", _norm(c))]
    for c in candidates:
        s = df[c]
        if s.notna().all() and s.astype(str).str.strip().ne("").all() and s.is_unique:
            return c
    return None


def detect_title_col(df, exclude=()):
    return _find(df, TITLE_HINTS, exclude)


def detect_post_col(df, exclude=()):
    hit = _find(df, POST_HINTS, exclude)
    if hit is not None:
        return hit
    # fallback: the text column with the longest average length
    obj = [c for c in df.columns if c not in exclude and df[c].dtype == "object"]
    if not obj:
        return None
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())


# ------------------------------------------------------------------ text cleaning
def clean_text(value, lowercase: bool = False) -> str:
    """
    Light, model-friendly cleaning. Keeps punctuation, CAPS and emojis because
    sentiment tools such as VADER use them. Set lowercase=True for classic ML models.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    t = str(value).strip()
    if t.lower() in PLACEHOLDERS:
        return ""

    t = re.sub(r"^\[\s*['\"]|['\"]\s*\]$", "", t)          # ['...'] list-string wrappers
    t = html.unescape(t)                                    # &amp; &gt; &#x200B; ...
    t = re.sub(r"<[^>]+>", " ", t)                          # html tags
    t = re.sub(r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)", r"\1", t)  # [text](url) -> text
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)            # bare URLs
    t = re.sub(r"\S+@\S+\.\S+", " ", t)                     # emails
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)        # zero-width characters
    t = re.sub(r"(\*\*|__|~~|`+|^>+\s*|^#+\s*)", " ", t, flags=re.M)  # markdown symbols
    t = re.sub(r"\\n|\\r|\\t", " ", t)                      # literal \n inside strings
    t = re.sub(r"([!?.,])\1{3,}", r"\1\1\1", t)             # !!!!!!!! -> !!!
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if lowercase else t


# ------------------------------------------------------------------ main entry point
def load_and_clean(path_or_df, id_col=None, title_col=None, post_col=None,
                   lowercase: bool = False, drop_duplicates: bool = True,
                   verbose: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with exactly three columns: post_id, title, post.

    Columns are auto-detected; pass id_col / title_col / post_col to override.
    """
    raw = load_dataset(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    n_start = len(raw)
    raw = raw.dropna(how="all").reset_index(drop=True)

    # --- choose columns
    id_col = id_col or detect_id_col(raw)
    used = [c for c in (id_col,) if c]
    title_col = title_col or detect_title_col(raw, exclude=used)
    used += [c for c in (title_col,) if c]
    post_col = post_col or detect_post_col(raw, exclude=used)
    if post_col is None and title_col is None:
        raise ValueError("No text column found. Pass post_col=... manually.")

    # --- build the 3-column frame
    out = pd.DataFrame()
    out["post_id"] = raw[id_col].astype(str).str.strip() if id_col else range(1, len(raw) + 1)
    out["title"] = raw[title_col].map(lambda v: clean_text(v, lowercase)) if title_col else ""
    out["post"] = raw[post_col].map(lambda v: clean_text(v, lowercase)) if post_col else ""

    # --- drop rows with no text at all
    empty = (out["title"] == "") & (out["post"] == "")
    n_empty = int(empty.sum())
    out = out[~empty]

    # --- drop duplicates (same id, or identical title+post)
    n_dup = 0
    if drop_duplicates:
        before = len(out)
        if id_col:
            out = out.drop_duplicates(subset="post_id")
        out = out.drop_duplicates(subset=["title", "post"])
        n_dup = before - len(out)

    # --- serial ids must stay 1..N after dropping rows? keep originals, they still identify rows
    out = out.reset_index(drop=True)

    if verbose:
        print("Cleaning report")
        print(f"  rows in file        : {n_start}")
        print(f"  id column           : {id_col or 'none found -> serial numbers 1..N'}")
        print(f"  title column        : {title_col or 'none found'}")
        print(f"  post column         : {post_col or 'none found'}")
        print(f"  dropped (no text)   : {n_empty}")
        print(f"  dropped (duplicates): {n_dup}")
        print(f"  rows kept           : {len(out)}")
    return out
