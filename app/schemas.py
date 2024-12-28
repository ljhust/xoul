from pydantic import BaseModel
from typing import Optional


class CompareRequest(BaseModel):
    text: str


class CompareResponse(BaseModel):
    id: int
    input_text: str
    response_model_base: str
    response_model_lora: str
    votes_model_base: int
    votes_model_lora: int


class StatisticsResponse(BaseModel):
    total_comparisons: int
    total_votes_model_base: int
    total_votes_model_lora: int

