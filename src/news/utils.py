from typing import Tuple, Any, Dict, Optional, List
from requests import get, Response
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl
from halo import Halo
from rich import print as rprint
from datetime import date
import pandas as pd


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    provider: str
    text: str
    sentiment: Optional[List[int]] = None
    predicted_sentiment: Optional[int] = None

    def __str__(self):
        return f"Title: {self.title}\nProvider: {self.provider}\nURL: {self.url}\nText: {self.text}"


def format_date(
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


def get_params(
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
        "tbs": format_date(start_date, end_date),
        "tbm": "nws",
        "start": start,
    }


def parse_and_populate(
    response: Response, results: List[SearchResult], count: int, end: int
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
        if count == end:
            break
        spinner = Halo(text=f"Scraping news article {count+1}", spinner="dots")
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
            count += 1
            spinner.stop()
        except Exception as e:
            spinner.fail()
            print(e)
            break
    return count


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


def get_daterange(
    start_date: Tuple[int, int, int], end_date: Tuple[int, int, int]
) -> List[Tuple[int, int, int]]:
    """
    Function to get the date range between two dates.

    Parameters
    ----------
    start_date : Tuple[int, int, int]
        A tuple of integers representing the start date. Format: (Month, Day, Year)
    end_date : Tuple[int, int, int]
        A tuple of integers representing the end date. Format: (Month, Day, Year)

    Returns
    -------
    List[Tuple[int, int, int]]
        A list of tuples representing the date range between the two dates.
    """
    start = date(start_date[2], start_date[0], start_date[1])
    end = date(end_date[2], end_date[0], end_date[1])
    dr = pd.date_range(start, end)
    return [(d.month, d.day, d.year) for d in dr]


def pprint_search_results(results: List[SearchResult]) -> None:
    """
    Function to pretty print the search results.

    Parameters
    ----------
    results : List[SearchResult]
        List of search results to pretty print.

    Returns
    -------
    None
    """
    for i, r in enumerate(results):
        rprint(f"[bold]Result {i+1}[/bold]")
        rprint("-" * 50)
        rprint(f"[red]Provider: {r.provider}[/red]")
        rprint(f"[blue]Title: {r.title}[/blue]")
        rprint(f"[green]URL: {r.url}[/green]")
        rprint(f"[yellow]Text: {r.text}[/yellow]")
        rprint("\n")
        rprint("\n")
