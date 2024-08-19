from typing import Tuple, List, Optional
from .utils import get_params
from .headers import get_random_agent
from .utils import parse_and_populate, SearchResult, get_daterange
from requests import get
from rich.progress import Progress


def get_news_for_date(
    ticker: str,
    date: Tuple[int, int, int],
    count: Optional[int] = 10,
    progress: Progress = None,
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
    parsed = 0
    if progress:
        day_task = progress.add_task(
            f"[cyan]Processing News for {date[0]}/{date[1]}/{date[2]}...",
            total=count,
        )
    else:
        progress = None
        day_task = None
    while parsed < count:
        try:
            params = get_params(ticker, date, date, parsed)
            response = get(url, params=params, headers=headers)
            parsed = parse_and_populate(
                response, results, parsed, count, progress, day_task
            )
        except Exception as e:
            print(f"News Fetch Failed: {e}")
            continue
        if parsed % 10 != 0:
            break
    if progress:
        progress.update(day_task, completed=count, visible=False)
    return results
