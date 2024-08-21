from nltk.sentiment.vader import SentimentIntensityAnalyzer
from finsent.indicators.news import SearchResult
from typing import List


class Vader:
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()

    def __get_sentiment(self, text: str) -> int:
        """
        Function to get the sentiment of a given text.

        Parameters
        ----------
        text : str
            The text to get the sentiment of.

        Returns
        -------
        int
            The sentiment score.
        """
        sentiment = self.model.polarity_scores(text)
        return sentiment["compound"]

    def process_search_result(self, sr: SearchResult) -> SearchResult:
        """
        Function to process a search result and add the sentiment to it.

        Parameters
        ----------
        sr : SearchResult
            The search result to process.

        Returns
        -------
        SearchResult
            The search result with the sentiment added.
        """
        sentiment = self.__get_sentiment(sr.text)
        sr.sentiment = sentiment
        return sr

    def process_list(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Function to process a list of search results and add the sentiment to them.

        Parameters
        ----------
        results : List[SearchResult]
            The list of search results to process.

        Returns
        -------
        List[SearchResult]
            The list of search results with the sentiment added.
        """
        for r in results:
            self.process_search_result(r)
        return results

    def process_list_for_sentiment(self, results: List[SearchResult]) -> float:
        """
        Function to process a list of search results and compute the daily average sentiment.

        Parameters
        ----------
        results : List[SearchResult]
            The list of search results to process.

        Returns
        -------
        float
            The daily average sentiment
        """
        return self.__compute_daily_averages(self.process_list(results))

    def __compute_daily_averages(self, results: List[SearchResult]) -> float:
        """
        Utility function to compute the daily average sentiment for a list of search results.

        Parameters
        ----------
        results : List[SearchResult]
            List of search results to compute the daily average sentiment for.

        Returns
        -------
        float
            The daily average sentiment
        """
        return sum([r.sentiment for r in results]) / len(results)
