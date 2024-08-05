from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from typing import List
from news.scrape import SearchResult


class SentimentAnalysis:
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("ProsusAI/finbert")
        self.model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

    def __chunk_and_tokenize(self, text: str) -> List[str]:
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
            probabilties = torch.nn.functional.softmax(outputs.logits, dim=-1)
            mean_prob = probabilties.mean(dim=0)
            sentiment = mean_prob.tolist()
            pred_sentiment = sentiment[0] - sentiment[2]
        return sentiment, pred_sentiment

    def process_search_result(self, sr: SearchResult):
        sentiment, pred_sentiment = self.__get_sentiment(sr.text)
        sr.sentiment = sentiment
        sr.predicted_sentiment = pred_sentiment
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
        total_neg += r.sentiment[2]
        total_neu += r.sentiment[1]

    avg_total = (total_pos - total_neg) / len(results)
    avg_pos = total_pos / len(results)
    avg_neg = total_neg / len(results)
    avg_neu = total_neu / len(results)
    return [(avg_pos, avg_neu, avg_neg), avg_total]
