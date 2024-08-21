from typing import List, Optional, Union, Literal
import pygwalker.api
import pygwalker.api.pygwalker
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from finsent.indicators import add_analyst_index, add_insider_index, add_news_index
from finsent.indicators.sentiment import SentimentAnalyzer
from dash import Dash
from finsent.utils import render_df
import pygwalker


class DataComposer:
    def __init__(
        self,
        ticker: str,
        load: Optional[str] = None,
        sentiment_analyzer: Optional[Literal["finbert", "vader"]] = None,
        preprocess_string_for_sentiment: bool = True,
    ):
        """
        Constructor for the DataComposer class.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.
        load : str
            The path to the file to load the data from.
        sentiment_analyzer : str
            The sentiment analyzer to use for processing the news. Valid values are "finbert" and "vader". Default is None. This property must be added if the "news" indicator is being used.
        preprocess_string_for_sentiment : bool
            Whether or not to preprocess the string before passing it to the sentiment analyzer. Default is True.
        """
        self.ticker = ticker
        if load:
            self.data: pd.DataFrame = pd.read_pickle(load)
            self.start_date = self.data["Date"].min().isoformat()
            self.end_date = self.data["Date"].max().isoformat()
        else:
            self.data: pd.DataFrame = None
            self.start_date = None
            self.end_date = None

        if sentiment_analyzer:
            self.sentiment_analyzer = SentimentAnalyzer(
                method=sentiment_analyzer, preprocess=preprocess_string_for_sentiment
            )
        else:
            self.sentiment_analyzer = None

    @property
    def data(self) -> pd.DataFrame:
        """Function to get the data attribute."""
        return self.data

    @property
    def sentiment_analyzer(self) -> SentimentAnalyzer:
        """Function to get the sentiment_analyzer attribute."""
        return self.sentiment_analyzer

    def add_update_sentiment_analyzer(self, method: str, preprocess: bool) -> None:
        """
        Function to add or update the sentiment analyzer.

        Parameters
        ----------
        method : str
            The sentiment analysis method to use. Valid values are "finbert" and "vader".
        preprocess : bool
            Whether or not to preprocess the string before passing it to the sentiment analyzer.
        """
        self.sentiment_analyzer = SentimentAnalyzer(
            method=method, preprocess=preprocess
        )

    def get_ticker_data(self, start_date: str, end_date: str) -> None:
        """
        Function to get the stock data for a given ticker and date range.

        Parameters
        ----------
        start_date : str
            The start date in the format "YYYY-MM-DD".
        end_date : str
            The end date in the format "YYYY-MM-DD".
        """
        tick = yf.Ticker(self.ticker)
        hist = tick.history(start=start_date, end=end_date)
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].dt.date
        self.df = hist[["Date", "Open", "Close", "High", "Low"]]
        self.start_date = start_date
        self.end_date = end_date

    def add(self, indicators: Union[str, List[str]]) -> None:
        """
        Function to add indicators to the stock data.

        The currently supported indicators are:
            - "news" which adds the "news" and "cumulative_news" columns.
            - "insider" which adds the "insider" column.
            - "analyst" which adds the "analyst" column.

        Parameters
        ----------
        indicators : str or List[str]
            The indicator(s) to add to the stock data.
        """
        if self.data is None:
            raise ValueError(
                "No data to add indicators to. Please load data first using get_ticker_data function call."
            )

        if isinstance(indicators, str):
            indicators = [indicators]

        for indicator in indicators:
            self.__redirect_indicator_additions(indicator)

    def __redirect_indicator_additions(self, indicator: str) -> None:
        """
        Function to redirect the addition of indicators to the appropriate method.

        Parameters
        ----------
        indicator : str
            The indicator to add to the stock data.
        """
        if indicator == "News":
            if self.sentiment_analyzer is None:
                raise ValueError(
                    "No sentiment analyzer provided. Please provide a sentiment analyzer to use for processing the news."
                )
            self.data = add_news_index(self.ticker, self.data, self.sentiment_analyzer)
        elif indicator == "Insider":
            self.data = add_insider_index(
                self.ticker, self.start_date, self.end_date, self.data
            )
        elif indicator == "Analyst":
            self.data = add_analyst_index(
                self.ticker, self.start_date, self.end_date, self.data
            )

    def save(self, path: str) -> None:
        """
        Function to save the stock data to a file.

        Parameters
        ----------
        path : str
            The path to save the stock data to.
        """
        self.data.to_pickle(path)

    def plot(self, misc_indicators: List[str] = None) -> None:
        """
        Function to plot the stock data. The stock data will be plotted as candlesticks along with any miscellaneous indicators provided.
        Miscellaneous indicators currently supported are:
            - "news"
            - "cumulative_news"
            - "insider"
            - "analyst"
        Miscellaneous indicators must be added to the stock data before plotting.
        Miscellaneous indicators will be plotted on a seperate y-axis.

        Parameters
        ----------
        misc_indicators : List[str]
            The miscellaneous indicators to plot along with the stock data.
        """
        if self.data is None:
            raise ValueError(
                "No data to plot. Please load data first using get_ticker_data function call."
            )

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=self.data["Date"],
                open=self.data["Open"],
                close=self.data["Close"],
                high=self.data["High"],
                low=self.data["Low"],
                name="Stock Data",
            )
        )

        if misc_indicators:
            for indicator in misc_indicators:
                self.__redirect_misc_indicator_plotting(indicator, fig)
            fig.update_layout(
                yaxis2={
                    "title": "Miscellaneous Indicators",
                    "overlaying": "y",
                    "side": "right",
                }
            )
        fig.update_layout(
            title=f"{self.ticker} Stock Data",
            xaxis={"title": "Date"},
            yaxis={"title": "Stock Price"},
        )
        fig.show()

    def __redirect_misc_indicator_plotting(
        self, indicator: str, fig: go.Figure
    ) -> None:
        """
        Function to redirect the plotting of miscellaneous indicators to the appropriate method.

        Parameters
        ----------
        indicator : str
            The miscellaneous indicator to plot.
        fig : go.Figure
            The plotly figure to add the indicator to.

        Returns
        -------
        go.Figure
            The plotly figure with the miscellaneous indicator added.
        """
        if indicator not in self.data.columns:
            raise ValueError(f"{indicator} not in stock data.")

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data[indicator],
                name=indicator,
                yaxis="y2",
                mode="lines+markers",
            )
        )

    def pygwalker_dashboard_dash(self) -> Dash:
        """
        Function to render the stock data as an Pygwalker HTML dashboard in Dash. This function is meant to be used while rendering the dashboard in terminal.

        This function uses the awesome Pygwalker library to render the stock data as an HTML dashboard in Dash with which you can do a lot of cool stuff like filtering, sorting, plotting, etc.

        To run the dashboard, you can use the following code:
        >>> dashboard = data_composer.pygwalker_dashboard_dash()
        >>> dashboard.run(debug=True)
        """
        return render_df(self.data)

    def pygwalker_dashboard(self) -> pygwalker.api.pygwalker.PygWalker:
        """
        Function to render the stock data as an Pygwalker HTML dashboard. This function is meant to be used while rendering the dashboard in a Jupyter Notebook.

        This function uses the awesome Pygwalker library to render the stock data as an HTML dashboard with which you can do a lot of cool stuff like filtering, sorting, plotting, etc.
        """
        return pygwalker.walk(self.data)
