from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.models import Comparison
from app.schemas import CompareResponse, StatisticsResponse


def create_comparison(db: Session, input_text: str, resp1: str, resp2: str) -> CompareResponse:
    comparison = Comparison(
        input_text=input_text,
        response_model_1=resp1,
        response_model_2=resp2,
    )
    db.add(comparison)
    db.commit()
    db.refresh(comparison)
    return CompareResponse(
        id=comparison.id,
        input_text=comparison.input_text,
        response_model_1=comparison.response_model_1,
        response_model_2=comparison.response_model_2,
        votes_model_1=comparison.votes_model_1,
        votes_model_2=comparison.votes_model_2,
    )


def cast_vote(db: Session, comparison_id: int, model_choice: str):
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise ValueError("Comparison not found.")

    if model_choice == "model_1":
        comparison.votes_model_1 += 1
    elif model_choice == "model_2":
        comparison.votes_model_2 += 1
        
    db.commit()


def get_statistics(db: Session) -> StatisticsResponse:
    total_comparisons = db.query(Comparison).count()
    total_votes_model_1 = db.query(Comparison).with_entities(
        func.sum(Comparison.votes_model_1)
    ).scalar() or 0
    total_votes_model_2 = db.query(Comparison).with_entities(
        func.sum(Comparison.votes_model_2)
    ).scalar() or 0

    return StatisticsResponse(
        total_comparisons=total_comparisons,
        total_votes_model_1=total_votes_model_1,
        total_votes_model_2=total_votes_model_2,
    )


