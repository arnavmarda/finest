from news.scrape import get_news_for_date
from news.utils import get_daterange
from sentiment.analysis import SentimentAnalysis
from typing import Tuple, List
from datetime import date


class NewsSentiment:
    def __init__(self):
        self.sa = SentimentAnalysis()

    def get_sentiment_score(
        self,
        ticker: str,
        start_date: Tuple[int, int, int],
        end_date: Tuple[int, int, int],
        count: int = 10,
    ) -> Tuple[List, List]:
        """
        Function to get the sentiment score per day for a given ticker and date range.

        Parameters
        ----------
        ticker : str
            The ticker symbol of the company.
        start_date : Tuple[int, int, int]
            The start date in the format (Month, Day, Year).
        end_date : Tuple[int, int, int]
            The end date in the format (Month, Day, Year).
        count : int
            The number of articles to consider for each day. Default is 10.

        Returns
        -------
        Tuple[List, List]
            The sentiment scores for each day in the date range. Cumulative sentiment score per day is also returned.
        """
        start_date = date(start_date[2], start_date[0], start_date[1])
        end_date = date(end_date[2], end_date[0], end_date[1])
        date_range = get_daterange(start_date, end_date)
        sentiment_score = []
        cum_sent_score = []
        curr_sent_score = 0
        for d in date_range:
            news = get_news_for_date(ticker, d, count)
            if not news:
                sentiment = [(0, 0, 0), 0]
            else:
                sentiment = self.sa.process_list_for_sentiment(news)

            curr_sent_score += sentiment[0]
            sentiment_score.append(tuple(sentiment))
            cum_sent_score.append(curr_sent_score)
        return sentiment_score, cum_sent_score
