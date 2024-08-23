from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class SearchResult(BaseModel):
    title: str
    url: HttpUrl
    provider: str
    text: str
    sentiment: Optional[List[int]] = None
    predicted_sentiment: Optional[int] = None

    def __str__(self):
        return f"Title: {self.title}\nProvider: {self.provider}\nURL: {self.url}\nText: {self.text}"
