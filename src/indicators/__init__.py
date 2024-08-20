import pandas as pd
from src.sentiment.analysis import SentimentAnalysis

import analyst
import insiders
import news


def add_analyst_index(
    ticker: str, start_date: str, end_date: str, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to calculate and add the analyst index to the existing stock data.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to calculate the analyst index for.
    start_date : str
        The start date to calculate the analyst index for. Format: "YYYY-MM-DD"
    end_date : str
        The end date to calculate the analyst index for. Format: "YYYY-MM-DD"
    data : pd.DataFrame
        The stock data to add the analyst index to.

    Returns
    -------
    pd.DataFrame
        The stock data with the analyst index added.
    """

    # Download the upgrades and downgrades data
    ud_data = analyst.__download_up_down_data(ticker)

    # Clean the upgrades and downgrades data
    ud_data = analyst.__clean_data(ud_data, start_date, end_date)

    # Merge the upgrades and downgrades data into the main DataFrame
    data = analyst.__merge_into_data_df(data, ud_data)

    return data


def add_insider_index(
    ticker: str, start_date: str, end_date: str, data: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to calculate and add the insider index to the existing stock data.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to calculate the insider index for.
    start_date : str
        The start date to calculate the insider index for. Format: "YYYY-MM-DD"
    end_date : str
        The end date to calculate the insider index for. Format: "YYYY-MM-DD"
    data : pd.DataFrame
        The stock data to add the insider index to.

    Returns
    -------
    pd.DataFrame
        The stock data with the insider index added.
    """
    insider_df = insiders.__get_data(ticker, start_date, end_date)

    if insider_df is None:
        return data

    data = insiders.__merge_into_data_df(data, insider_df)
    data.rename(columns={"Amount": "InsiderIndex"}, inplace=True)

    return data


def add_news_index(
    ticker: str,
    data: pd.DataFrame,
    sa: SentimentAnalysis,
    count: int = 10,
) -> pd.DataFrame:
    """
    Function to calculate and add the news sentiment to the existing stock data.

    Parameters
    ----------
    ticker : str
        The ticker for the stock to calculate the news sentiment for.
    start_date : str
        The start date to calculate the news sentiment for. Format: "YYYY-MM-DD"
    end_date : str
        The end date to calculate the news sentiment for. Format: "YYYY-MM-DD"
    data : pd.DataFrame
        The stock data to add the news sentiment to.
    count : int
        The number of articles to consider for each day. Default is 10.

    Returns
    -------
    pd.DataFrame
        The stock data with the news sentiment added.
    """
    dates = data["Date"]
    dates = [(d.month, d.day, d.year) for d in dates]
    news_data = news.__get_sentiment_score(ticker, dates, sa, count)
    data = news.__merge_data_into_df(data, news_data)
    return data
