from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from typing import List
from finsent.indicators.news import SearchResult
import warnings
import os
import sys


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class FinBert:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
        # Mute the warnings from the tokenizer.
        self.tokenizer.model_max_length = sys.maxsize

    def __chunk_and_tokenize(self, text: str) -> List[str]:
        """
        Function to chunk and tokenize the text to fit the 512 token limit.

        Parameters
        ----------
        text : str
            The text to chunk and tokenize.

        Returns
        -------
        List[str]
            A list of tokenized text chunks
        """
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
        """
        Function to process the text and get the sentiment using FinBert.

        Parameters
        ----------
        text : str
            The text to process.

        Returns
        -------
        int
            The sentiment score.
        """
        tokens = self.__chunk_and_tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)
            probabilties = torch.nn.functional.softmax(outputs[0], dim=-1)
            mean_prob = probabilties.mean(dim=0)
            sentiment = mean_prob.tolist()
        return sentiment

    def process_search_result(self, sr: SearchResult):
        """
        Function to process a single search result and add the sentiment to it.

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

    def __compute_daily_averages(results: List[SearchResult]) -> float:
        """
        Utility function to compute the daily average sentiment for a list of search results.

        Parameters
        ----------
        results : List[SearchResult]
            List of search results to compute the daily average sentiment for.

        Returns
        -------
        float
            The daily average sentiment.
        """
        total_neg = 0
        total_pos = 0
        for r in results:
            total_pos += r.sentiment[0]
            total_neg += r.sentiment[1]

        avg_total = (total_pos - total_neg) / len(results)
        return avg_total
