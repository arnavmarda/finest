import random
from typing import Tuple, Any, Dict, Optional, List
from requests import get, Response
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl
from datetime import date
import pandas as pd
from rich.progress import Progress, TaskID
from interruptingcow import timeout
from src.sentiment.analysis import SentimentAnalysis
import warnings


####################################################################################################
###################################### UTILITY FUNCTIONS ###########################################
####################################################################################################

agent_list = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 Edg/111.0.1661.62",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/111.0",
]


def __get_random_agent():
    """
    Function to get a random user agent for requests library for scraping.

    Returns
    -------
    str
        A random user agent string.
    """
    return random.choice(agent_list)


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    provider: str
    text: str
    sentiment: Optional[List[int]] = None
    predicted_sentiment: Optional[int] = None

    def __str__(self):
        return f"Title: {self.title}\nProvider: {self.provider}\nURL: {self.url}\nText: {self.text}"


def __format_date(
    start_date: Tuple[int, int, int], end_date: Tuple[int, int, int]
) -> str:
    """
    Function to format the start and end date for the URL.

    Parameters
    ----------
    start_date : Tuple[int, int, int]
        A tuple of integers representing the start date. Format: (Month, Day, Year)
    end_date : Tuple[int, int, int]
        A tuple of integers representing the end date. Format: (Month, Day, Year)

    Returns
    -------
    str
        A formatted string of the start and end date.
    """
    start_date_str = f"cd_min:{start_date[0]}/{start_date[1]}/{start_date[2]}"
    end_date_str = f"cd_max:{end_date[0]}/{end_date[1]}/{end_date[2]}"

    return f"cdr:1,{start_date_str},{end_date_str}"


def __get_params(
    ticker: str,
    start_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    start: int,
) -> Dict["str", Any]:
    """
    Function to get the parameters for the URL.

    Parameters
    ----------
    ticker : str
        A string representing the stock ticker.
    start_date : Tuple[int, int, int]
        A tuple of integers representing the start date. Format: (Month, Day, Year)
    end_date : Tuple[int, int, int]
        A tuple of integers representing the end date. Format: (Month, Day, Year)
    start : int
        An integer representing the start index for the search.

    Returns
    -------
    Dict[str, Any]
        A dictionary of parameters for the URL.
    """
    return {
        "q": ticker,
        "tbs": __format_date(start_date, end_date),
        "tbm": "nws",
        "start": start,
    }


def __parse_and_populate(
    response: Response,
    results: List[SearchResult],
    count: int,
    end: int,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """
    Function to parse and populate the list with results from the particular google search page.

    Parameters
    ----------
    response : requests.Response
        Response from GET request to google.
    results : List[SearchResult]
        List to append new found data to.
    count : int
        The number of news articles scraped so far.
    end : int
        The number of news articles to scrape.

    Returns
    -------
    None
    """
    soup = BeautifulSoup(response.text, "html.parser")
    blocks = soup.find_all("div", attrs={"class": "SoaBEf"})
    for b in blocks:
        if count >= end:
            break
        title = b.find("div", attrs={"aria-level": "3", "role": "heading"}).text
        link = b.find("a", href=True).get("href")
        provider = b.find("div", attrs={"class": "MgUUmf NUnG9d"}).text
        try:
            with warnings.catch_warnings(action="ignore"):
                with timeout(5, exception=RuntimeError):
                    resp = get(link)
                    text = __clean_webpage(resp)
            results.append(
                SearchResult(title=title, url=link, provider=provider, text=text)
            )
            count += 1
            if progress:
                progress.update(task_id, advance=1)
        except RuntimeError:
            count += 1
            if progress:
                progress.update(task_id, advance=1)
            continue
        except Exception as e:
            print(e)
            break
    return count


def __clean_webpage(response: Response) -> str:
    """
    Function to clean the webpage text.

    Parameters
    ----------
    response : requests.Response
        Response from GET request to google.

    Returns
    -------
    str
        Cleaned text from the webpage.
    """
    soup = BeautifulSoup(response.text, "html.parser")
    for data in soup(["style", "script"]):
        # Remove tags
        data.decompose()

    # return data by retrieving the tag content
    return " ".join(soup.stripped_strings)


def __get_news_for_date(
    ticker: str,
    search_date: Tuple[int, int, int],
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
        "User-Agent": __get_random_agent(),
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
            params = __get_params(ticker, search_date, search_date, parsed)
            response = get(url, params=params, headers=headers)
            parsed = __parse_and_populate(
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


def __get_sentiment_score(
    self,
    ticker: str,
    date_range: List[Tuple],
    sa: SentimentAnalysis,
    count: int = 10,
) -> List[float]:
    """
    Function to get the sentiment score per day for a given ticker and date range.

    Parameters
    ----------
    ticker : str
        The ticker symbol of the company.
    date_range : List[Tuple]
        The date range in the format [(Month, Day, Year), ...].
    count : int
        The number of articles to consider for each day. Default is 10.

    Returns
    -------
    List[float]
        A list of sentiment scores for each day in the date range.
    """
    sentiment_score = []

    with Progress() as progress:
        main_task = progress.add_task(
            "[cyan]Processing Sentiment Scores...", total=len(date_range)
        )
        for d in date_range:
            news = __get_news_for_date(ticker, d, count, progress)
            if not news:
                sentiment = [0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sentiment = sa.process_list_for_sentiment(news)

            sentiment_score.append(sentiment[1])
            progress.update(main_task, advance=1)
        return sentiment_score


def __merge_data_into_df(
    data: pd.DataFrame,
    news_data: List[float],
) -> pd.DataFrame:
    """
    Function to merge the news data into the main DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        The main DataFrame containing the stock data.
    news_data : pd.DataFrame
        The DataFrame containing the news data.

    Returns
    -------
    pd.DataFrame
        The main DataFrame with the news data merged.
    """

    data["NewsSentiment"] = news_data
    data["CumulativeNewsSentiment"] = data["NewsSentiment"].cumsum()
    return data
