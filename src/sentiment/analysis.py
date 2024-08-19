from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from typing import List
from news.scrape import SearchResult
import warnings
import os
import sys

# TODO: Add stopword removal, and stemming to improve the sentiment analysis.
# TODO: Add Loughran-McDonald dictionary to improve the sentiment analysis.


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class SentimentAnalysis:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.tokenizer.model_max_length = sys.maxsize

    def __chunk_and_tokenize(self, text: str) -> List[str]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                tokens = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
        chunksize = 512
        input_id_chunks = list(tokens["input_ids"][0].split(chunksize - 2))
        attention_mask_chunks = list(tokens["attention_mask"][0].split(chunksize - 2))

        for i in range(len(input_id_chunks)):
            input_id_chunks[i] = torch.cat(
                [torch.tensor([101]), input_id_chunks[i], torch.tensor([102])]
            )

            attention_mask_chunks[i] = torch.cat(
                [torch.tensor([1]), attention_mask_chunks[i], torch.tensor([1])]
            )

            pad_length = chunksize - input_id_chunks[i].shape[0]

            if pad_length > 0:
                input_id_chunks[i] = torch.cat(
                    [input_id_chunks[i], torch.Tensor([0] * pad_length)]
                )
                attention_mask_chunks[i] = torch.cat(
                    [attention_mask_chunks[i], torch.Tensor([0] * pad_length)]
                )
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(attention_mask_chunks)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.int(),
        }

    def __get_sentiment(self, text: str) -> int:
        tokens = self.__chunk_and_tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)
            probabilties = torch.nn.functional.softmax(outputs[0], dim=-1)
            mean_prob = probabilties.mean(dim=0)
            sentiment = mean_prob.tolist()
        return sentiment

    def process_search_result(self, sr: SearchResult):
        sentiment = self.__get_sentiment(sr.text)
        sr.sentiment = sentiment
        return sr

    def process_list(self, results: List[SearchResult]) -> List[SearchResult]:
        for r in results:
            self.process_search_result(r)
        return results

    def process_list_for_sentiment(self, results: List[SearchResult]) -> List:
        return compute_daily_averages(self.process_list(results))


def compute_daily_averages(results: List[SearchResult]) -> List:
    """
    Utility function to compute the daily average sentiment for a list of search results.

    Parameters
    ----------
    results : List[SearchResult]
        List of search results to compute the daily average sentiment for.

    Returns
    -------
    List
        List containing the sentiment scores and the total average sentiment.
    """
    total_neg = 0
    total_pos = 0
    total_neu = 0
    for r in results:
        total_pos += r.sentiment[0]
        total_neg += r.sentiment[1]
        total_neu += r.sentiment[2]

    avg_total = (total_pos - total_neg) / len(results)
    avg_pos = total_pos / len(results)
    avg_neg = total_neg / len(results)
    avg_neu = total_neu / len(results)
    return [(avg_pos, avg_neu, avg_neg), avg_total]
