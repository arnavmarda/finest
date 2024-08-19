from news.scrape import get_news_for_date
from news.utils import get_daterange
from sentiment.analysis import SentimentAnalysis
from typing import Tuple, List
from datetime import date
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from rich.progress import Progress
import warnings
import numpy as np
import os


class NewsSentiment:
    def __init__(self):
        self.sa = SentimentAnalysis()

    def get_sentiment_score(
        self,
        ticker: str,
        date_range: List[Tuple],
        count: int = 10,
    ) -> Tuple[List, List]:
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
        Tuple[List, List]
            The sentiment scores for each day in the date range. Cumulative sentiment score per day is also returned.
        """
        sentiment_score = []
        cum_sent_score = []
        curr_sent_score = 0
        with Progress() as progress:
            main_task = progress.add_task(
                "[cyan]Processing Sentiment Scores...", total=len(date_range)
            )
            for d in date_range:
                news = get_news_for_date(ticker, d, count, progress)
                if not news:
                    sentiment = [0]
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sentiment = self.sa.process_list_for_sentiment(news)

                curr_sent_score += sentiment[1]
                sentiment_score.append(sentiment[1])
                cum_sent_score.append(curr_sent_score)
                progress.update(main_task, advance=1)
            return sentiment_score, cum_sent_score

    def __get_ticker_data(
        self,
        ticker: str,
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
    ) -> List:
        """
        Function to get the stock data for a given ticker and date range.
        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.
        start_date : Tuple[int, int, int]
            The start date in the format (Month, Day, Year).
        end_date : Tuple[int, int, int]
            The end date in the format (Month, Day, Year).

        Returns
        -------
        Pandas DataFrame
            The stock data for the given ticker and date range.
        """
        tick = yf.Ticker(ticker)
        hist = tick.history(
            start=f"{start_date[2]}-{start_date[0]}-{start_date[1]}",
            end=f"{end_date[2]}-{end_date[0]}-{end_date[1]}",
        )
        hist.reset_index(inplace=True)
        return hist[["Date", "Open", "Close", "High", "Low"]]

    def __plot_stock_sentiment_data(
        self,
        stock_data: pd.DataFrame,
    ) -> None:
        """
        Function to plot the stock data and sentiment score data.

        Parameters
        ----------
        sentiment_score : List
            The sentiment scores for each day in the date range.
        cum_sent_score : List
            The cumulative sentiment scores for each day in the date range.
        stock_data : Pandas DataFrame
            The stock data for the given ticker and date range.
        """
        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=stock_data["Date"],
                open=stock_data["Open"],
                close=stock_data["Close"],
                high=stock_data["High"],
                low=stock_data["Low"],
                name="Stock Data",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Sentiment Score"],
                name="Daily Sentiment Score",
                yaxis="y2",
                mode="lines+markers",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=stock_data["5 Day Rolling Avg Sentiment Score"],
                name="5 Day Rolling Avg Sentiment Score",
                yaxis="y2",
                mode="lines+markers",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=stock_data["Cumulative Sentiment Score"],
                name="Cumulative Sentiment Score",
                yaxis="y2",
                mode="lines+markers",
            )
        )

        fig.update_layout(
            title="Stock Data and Sentiment Score",
            xaxis={"title": "Date"},
            yaxis={
                "title": "Stock Price",
            },
            yaxis2={"title": "Sentiment Score", "overlaying": "y", "side": "right"},
        )
        fig.show()

    def __consolidate(
        self,
        sentiment_score: List,
        cum_sent_score: List,
        stock_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Function to consolidate the stock data and sentiment score data into a single DataFrame.

        Parameters
        ----------
        sentiment_score : List
            The sentiment scores for each day in the date range.
        cum_sent_score : List
            The cumulative sentiment scores for each day in the date range.
        stock_data : Pandas DataFrame
            The stock data for the given ticker and date range.

        Returns
        -------
        Pandas DataFrame
            The consolidated DataFrame containing the stock data and sentiment score data.
        """
        stock_data["Sentiment Score"] = sentiment_score
        stock_data["5 Day Rolling Avg Sentiment Score"] = (
            stock_data["Sentiment Score"].rolling(5, min_periods=1).mean()
        )
        stock_data["Cumulative Sentiment Score"] = cum_sent_score
        return stock_data

    def __load_from_cache(self, ticker: str) -> pd.DataFrame | None:
        """
        Function to load the stock data and sentiment score data from the cache.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.

        Returns
        -------
        Pandas DataFrame
            The stock data and sentiment score data for the given ticker.
        """
        if os.path.exists(f"./saves/{ticker}.pkl"):
            return pd.read_pickle(f"./saves/{ticker}.pkl")
        return None

    def __save_to_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """
        Function to save the stock data and sentiment score data to the cache.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.
        df : Pandas DataFrame
            The stock data and sentiment score data to save.
        """
        if os.path.exists(f"./saves/{ticker}.pkl"):
            # Merge the new data with the old data
            old_df = pd.read_pickle(f"./saves/{ticker}.pkl")
            df = pd.merge(old_df, df, how="outer", on="Date")

            # Remove old data
            os.remove(f"./saves/{ticker}.pkl")

        df.to_pickle(f"./saves/{ticker}.pkl")

    def __check_if_data_in_df(
        self,
        df: pd.DataFrame,
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
    ) -> bool:
        """
        Function to check if the data in the DataFrame is up to date.

        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame to check.
        start_date : Tuple[int, int, int]
            The start date in the format (Month, Day, Year).
        end_date : Tuple[int, int, int]
            The end date in the format (Month, Day, Year).

        Returns
        -------
        bool
            True if the data in the DataFrame is up to date, False otherwise.
        """
        start_date = f"{start_date[2]}-{start_date[0]}-{start_date[1]}"
        end_date = f"{end_date[2]}-{end_date[0]}-{end_date[1]}"

        if len(df.query("@start_date <= Date <= @end_date")) == 0:
            return False
        return True

    def get_news_sentiment_and_plot(
        self,
        ticker: str,
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        count: int = 10,
        save: bool = True,
        load_from_cache: bool = True,
    ) -> None:
        """
        Function to get the sentiment score per day for a given ticker and date range and plot the stock data and sentiment score data.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.
        start_date : Tuple[int, int, int]
            The start date in the format (Month, Day, Year). Inclusive.
        end_date : Tuple[int, int, int]
            The end date in the format (Month, Day, Year). Exclusive.
        count : int
            The number of articles to consider for each day. Default is 10.
        save : bool
            Whether to save the data to the cache. Default is True.
        load_from_cache : bool
            Whether to load the data from the cache. Default is True.

        """
        loaded = False
        if load_from_cache:
            df = self.__load_from_cache(ticker)

            # Check if data in cache is up to date
            if df is not None and not self.__check_if_data_in_df(
                df, start_date, end_date
            ):
                df = None

        if df is not None:
            loaded = True

        if df is None:
            stock_data = self.__get_ticker_data(ticker, start_date, end_date)
            dates = stock_data["Date"].dt.date
            date_range = [(d.month, d.day, d.year) for d in dates]
            sentiment_score, cum_sent_score = self.get_sentiment_score(
                ticker, date_range, count
            )
            df = self.__consolidate(sentiment_score, cum_sent_score, stock_data)

        if save and not loaded:
            self.__save_to_cache(ticker, df)

        self.__plot_stock_sentiment_data(df)
