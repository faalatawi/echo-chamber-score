import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

from emoji import is_emoji

from typing import Optional
import logging
import string


# ----------------- #
# Constants
try:
    STOP_WORDS = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")  # download the stopwords corpus
    STOP_WORDS = stopwords.words("english")

PUNCTUATION = string.punctuation
PUNCTUATION += "“’”‘"

# Change this to change the tokens
TOKEN_MAP = {
    "user": "@user",
    "url": "http",
}


# ----------------- #
# Tweet level functions
def is_retweet(tweet: str) -> bool:
    return tweet.startswith("RT")


def is_quote(tweet: str) -> bool:
    # return true is tweet contains QT
    return "QT" in tweet


# ----------------- #
# Token level functions
def is_user(token: str) -> bool:
    return token.startswith("@") and len(token) > 1


def is_url(token: str) -> bool:
    # TODO use regex to check if token is a url in the future
    return token.startswith("http") or token.startswith("www")


def is_hashtag(token: str) -> bool:
    return token.startswith("#")


def is_number(token: str) -> bool:
    return token[0].isdigit()


# ----------------- #
# Cleaning functions:
def remove_leading_users(tweet_tokens: list[str]) -> list[str]:
    # remove leading users
    while len(tweet_tokens) > 0 and tweet_tokens[0].startswith("@"):
        tweet_tokens.pop(0)
    return tweet_tokens


def remove_stopwords(tweet_tokens: list[str], stopwords: list[str]) -> list[str]:
    return [word for word in tweet_tokens if word not in stopwords]


def remove_punctuation(tweet_tokens: list[str], punctuation: str) -> list[str]:
    return [word for word in tweet_tokens if word not in punctuation]


def remove_urls(tweet_tokens: list[str]) -> list[str]:
    return [word for word in tweet_tokens if not is_url(word)]


# Replace functions:
def replace_urls(tokens: list[str]) -> list[str]:
    return [TOKEN_MAP["url"] if is_url(token) else token for token in tokens]


def replace_users(tokens: list[str]) -> list[str]:
    return [TOKEN_MAP["user"] if is_user(token) else token for token in tokens]


def expend_contraction(word: str) -> str:
    word = word.lower()
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "it's": "it is",
        "i'm": "i am",
        "i've": "i have",
    }
    return contractions[word]


# ----------------- #
# ----------------- #
# ----------------- #
# Public functions:
# ----------------- #
# ----------------- #
# ----------------- #


def preprocess_tweet_for_lda(
    tweet: str,
    stop_words: list[str] = STOP_WORDS,
    punctuation: str = PUNCTUATION,
    tokenizer: Optional[TweetTokenizer] = None,
    lemmatizer: Optional[WordNetLemmatizer] = None,
    __debug: bool = False,
) -> list[str]:
    """
    Basic preprocessing for tweets
    Good for LDA topic modeling, TI-IDF, and other similar text analysis

    Type 1 preprocessing:
    - lowercase
    - remove leading users
    - remove stopwords
    - remove punctuation
    - remove numbers
    - remove `RT` and `QT`
    - remove the `#` from hashtags
    - remove emoji
    - remove urls
    - remove user
    - lemmatize
    """
    if tokenizer is None:
        tokenizer = TweetTokenizer()

    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

    if tweet.startswith("Wordle"):
        if __debug:
            logging.info(f"Skipping Wordle tweet: {tweet}")

        return []

    tweet = tweet.lower()

    # Remove flag emojis (This is a hacky way to avoid problem with TweetTokenizer)
    tweet = tweet.replace("🇺🇸", " us ")
    tweet = tweet.replace("🇺🇦", " ua ")
    tweet = tweet.replace("🇷🇺", " ru ")

    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_tokens = remove_leading_users(tweet_tokens)

    output_tokens = []
    for tok in tweet_tokens:
        if tok in stop_words or tok in punctuation:
            continue
        elif tok in ["rt", "qt"]:
            continue
        elif is_number(tok) or is_user(tok) or is_url(tok):
            continue
        elif is_emoji(tok):
            continue
        elif is_hashtag(tok):
            output_tokens.append(tok[1:])
        else:
            tok = lemmatizer.lemmatize(tok)
            output_tokens.append(tok)

    return output_tokens


def preprocess_tweet_for_bert(
    tweet: str,
    tokenizer: Optional[TweetTokenizer] = None,
    token_map=TOKEN_MAP,
    __debug: bool = False,
) -> list[str]:
    """
    preprocessing for tweets for BERT

    Type 1 preprocessing:
    - lowercase
    - remove leading users
    - remove `RT` and `QT`
    - remove the `#` from hashtags ?
    - replace emoji with text
    - replace urls with 'http'
    - replace users with '@user'
    """
    if tokenizer is None:
        tokenizer = TweetTokenizer()

    if tweet.startswith("Wordle"):
        if __debug:
            logging.info(f"Skipping Wordle tweet: {tweet}")
        return []

    tweet = tweet.lower()

    tweet_tokens = tokenizer.tokenize(tweet)
    tweet_tokens = remove_leading_users(tweet_tokens)

    output_tokens = []
    for tok in tweet_tokens:
        if tok in ["rt", "qt"]:
            continue
        elif is_url(tok):
            output_tokens.append(token_map["url"])
        elif is_user(tok):
            output_tokens.append(token_map["user"])
        elif is_hashtag(tok):
            # TODO: Not sure if we should remove the hashtag `#`, Maybe we should handle it better
            output_tokens.append(
                tok[1:]
            )  # ? Not sure if we should remove the hashtag `#`
        else:
            output_tokens.append(tok)

    return output_tokens
