from requests import get, Response
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl
from typing import Tuple, List, Optional
from .utils import get_params
from .headers import get_random_agent
from halo import Halo


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    provider: str
    text: str
    sentiment: Optional[List[int]] = None
    predicted_sentiment: Optional[int] = None


def get_news_for_date(ticker: str, date: Tuple[int, int, int]) -> List[SearchResult]:
    """
    Function to get financial news for ticker for a particular date.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to scrape financial data for.
    date : Tuple[int, int, int]
        The date to get the financial data for. Format: [Month, Day, Year]

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

    for start in [0, 10, 20]:
        try:
            params = get_params(ticker, date, date, start)
            response = get(url, params=params, headers=headers)
            parse_and_populate(response, results)
        except Exception:
            continue
    return results


def parse_and_populate(response: Response, results: List[SearchResult]) -> None:
    """
    Function to parse and populate the list with results from the particular google search page.

    Parameters
    ----------
    response : requests.Response
        Response from GET request to google.
    results : List[SearchResult]
        List to append new found data to.

    Returns
    -------
    None
    """
    soup = BeautifulSoup(response.text, "html.parser")
    blocks = soup.find_all("div", attrs={"class": "SoaBEf"})
    for b in blocks:
        spinner = Halo(text=f"Scraping news article {len(results)+1}", spinner="dots")
        spinner.start()
        title = b.find("div", attrs={"aria-level": "3", "role": "heading"}).text
        link = b.find("a", href=True).get("href")
        provider = b.find("div", attrs={"class": "MgUUmf NUnG9d"}).text
        try:
            resp = get(link)
            text = clean_webpage(resp)
            results.append(
                SearchResult(title=title, url=link, provider=provider, text=text)
            )
            spinner.stop()
        except Exception as e:
            spinner.fail()
            print(e)
            break


def clean_webpage(response: Response) -> str:
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
