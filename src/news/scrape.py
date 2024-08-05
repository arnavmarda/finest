from typing import Tuple, List, Optional
from .utils import get_params
from .headers import get_random_agent
from .utils import parse_and_populate, SearchResult, get_daterange
from requests import get


def get_news_for_date(
    ticker: str, date: Tuple[int, int, int], count: Optional[int] = 10
) -> List[SearchResult]:
    """
    Function to get financial news for ticker for a particular date.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to scrape financial data for.
    date : Tuple[int, int, int]
        The date to get the financial data for. Format: [Month, Day, Year]
    count : Optional[int]
        The number of news articles to scrape. Default is 10.

    Returns
    -------
    List[SearchResult]
        List of results from scraping google. Results include title, url, provider, text.
    """
    headers = {
        "User-Agent": get_random_agent(),
    }
    url = "https://www.google.com/search"
    results = []
    parsed = 1
    start = 0
    while parsed < count:
        start += 10
        try:
            params = get_params(ticker, date, date, start)
            response = get(url, params=params, headers=headers)
            parsed = parse_and_populate(response, results, parsed, count)
        except Exception:
            continue
    return results
