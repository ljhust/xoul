from pydantic import BaseModel
from typing import Optional


class CompareRequest(BaseModel):
    text: str


class CompareResponse(BaseModel):
    id: int
    input_text: str
    response_model_1: str
    response_model_2: str
    votes_model_1: int
    votes_model_2: int


class StatisticsResponse(BaseModel):
    total_comparisons: int
    total_votes_model_1: int
    total_votes_model_2: int

