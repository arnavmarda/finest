from src.indicators.sentiment.finbert import FinBert
from src.indicators.sentiment.vader import Vader
from typing import Literal, List
from src.indicators.news import SearchResult

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from string import punctuation


class SentimentAnalyzer:
    def __init__(
        self, method: Literal["finbert", "vader"] = "finbert", preprocess: bool = True
    ):
        if method == "finbert":
            self.analyzer = FinBert()
        else:
            self.analyzer = Vader()

        if preprocess:
            self.stemmer = PorterStemmer()
            nltk.download("stopwords")
            self.stopwords = set(stopwords.words("english"))

    def __preprocess_text(self, text: str) -> str:
        """
        Function to preprocess the text by removing stopwords, punctuation, and stemming.

        Parameters
        ----------
        text : str
            The text to preprocess.

        Returns
        -------
        str
            The preprocessed text.
        """
        # Lowercase the text
        text = text.lower()

        # Remove punctuation (except for periods, to avoid sentence merging)
        text = re.sub(r"[{}]".format(re.escape(punctuation)), " ", text)

        # Split the text into words
        words = text.split()

        # Remove stop words and stem the words
        processed_words = [
            self.stemmer.stem(word) for word in words if word not in self.stop_words
        ]

        # Join the processed words back into a single string
        processed_text = " ".join(processed_words)

        # Remove redundant spaces and return the processed text
        processed_text = re.sub(r"\s+", " ", processed_text).strip()

        return processed_text

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
        if self.preprocess:
            sr.text = self.__preprocess_text(sr.text)
        return self.analyzer.process_search_result(sr)

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
        if self.preprocess:
            for r in results:
                r.text = self.__preprocess_text(r.text)
        return self.analyzer.process_list(results)

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
        if self.preprocess:
            for r in results:
                r.text = self.__preprocess_text(r.text)
        return self.analyzer.process_list_for_sentiment(results)
