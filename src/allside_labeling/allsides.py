from urllib.parse import urlparse
import pandas as pd
from collections import namedtuple


MISSING_OR_MISTAKES_URL = {
    "nyti.ms": "nytimes.com",
    "wapo.st": "washingtonpost.com",
    "abcn.ws": "abcnews.go.com",
    "fxn.ws": "foxnews.com",
    "n.pr": "npr.org",
    "nbcnews.to": "nbcnews.com",
    "ti.me": "time.com",
    "reut.rs": "reuters.com",
    "st.news": "seattletimes.com",
    "cnn.it": "cnn.com",
    "vntyfr.com": "vanityfair.com",
    "nyer.cm": "newyorker.com",
    "tnvge.co": "teenvogue.com",
    "econ.st": "economist.com",
    "cbsn.ws": "cbsnews.com",
    "rol.st": "rollingstone.com",
    "slate.trib.al": "slate.com",
    # Error with the url
    "on.wsj.com": "wsj.com",
    "to.pbs.org": "pbs.org",
    "on.msnbc.com": "msnbc.com",
    "news.yahoo.com": "yahoo.com",
    "businessinsider.com": "insider.com",
}


def build_allsides_rating_dict(allsides_path: str) -> dict[str, float]:

    df_allsides = pd.read_csv(allsides_path)

    # Keep only the domain and the rating
    df_allsides = df_allsides[["domain", "rating"]]

    # map ratings to numbers
    rating_map = {
        "left": -1.0,
        "left_center": -0.5,
        "center": 0.0,
        "right_center": 0.5,
        "right": 1.0,
    }

    df_allsides["rating"] = df_allsides["rating"].map(rating_map)

    # Get the dict
    domain2rating = df_allsides.set_index("domain").to_dict()["rating"]

    # Add missing URLs to domain2rating
    for key, value in MISSING_OR_MISTAKES_URL.items():
        if key not in domain2rating:
            domain2rating[key] = domain2rating[value]

    return domain2rating


def get_hostname(url):
    return urlparse(url).hostname.replace("www.", "")


def get_allsides_ideology_score(
    df_user: pd.DataFrame, domain2rating: dict[str, float]
) -> float:
    """
    Only if users has outLinks column
    """

    URLs = df_user.outlinks.dropna().tolist()
    URLs_flat = []
    for url in URLs:
        URLs_flat.extend(url)
    URLs = URLs_flat

    domain_score = []

    for url in URLs:
        hostname = get_hostname(url)
        if hostname in domain2rating:
            domain_score.append(domain2rating[hostname])

    if len(domain_score) == 0:
        return float("nan")

    return sum(domain_score) / len(domain_score)


allsides_user = namedtuple(
    "allsides_user",
    [
        "user_id",
        "allsides_score",
        "outlinks_len",
        "used_links_len",
        "apple_or_msn_urls",
    ],
)


def get_allsides_ideology_object(
    user_id, df_user: pd.DataFrame, domain2rating: dict[str, float]
) -> allsides_user:
    """
    Only if users has outLinks column
    """

    URLs = df_user.outlinks.dropna().tolist()
    URLs_flat = []
    for url in URLs:
        URLs_flat.extend(url)
    URLs = URLs_flat

    domain_score = []
    apple_or_msn_urls = []

    for url in URLs:
        hostname = get_hostname(url)
        if hostname in domain2rating:
            domain_score.append(domain2rating[hostname])

        if hostname in ["apple.news", "msn.com"]:
            apple_or_msn_urls.append(url)

    if len(domain_score) == 0:
        return allsides_user(
            user_id, float("nan"), len(URLs), len(domain_score), apple_or_msn_urls
        )

    return allsides_user(
        user_id,
        sum(domain_score) / len(domain_score),
        len(URLs),
        len(domain_score),
        apple_or_msn_urls,
    )
