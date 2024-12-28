from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.models import Comparison
from app.schemas import CompareResponse, StatisticsResponse

# Function to create a new comparison entry in the database
def create_comparison(db: Session, input_text: str, resp1: str, resp2: str) -> CompareResponse:
    # Create a new Comparison object
    comparison = Comparison(
        input_text=input_text,
        response_model_base=resp1,
        response_model_lora=resp2,
    )
    # Add the new comparison to the session
    db.add(comparison)
    # Commit the session to save the comparison to the database
    db.commit()
    # Refresh the session to reflect the new state of the comparison
    db.refresh(comparison)
    # Return a response object with the comparison details
    return CompareResponse(
        id=comparison.id,
        input_text=comparison.input_text,
        response_model_base=comparison.response_model_base,
        response_model_lora=comparison.response_model_lora,
        votes_model_base=comparison.votes_model_base,
        votes_model_lora=comparison.votes_model_lora,
    )

# Function to cast a vote for a specific model in a comparison
def cast_vote(db: Session, comparison_id: int, model_choice: str):
    # Query the database for the comparison with the given ID
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    # Raise an error if the comparison is not found
    if not comparison:
        raise ValueError("Comparison not found.")

    # Increment the vote count for the chosen model
    if model_choice == "model_base":
        comparison.votes_model_1 += 1
    elif model_choice == "model_lora":
        comparison.votes_model_2 += 1
        
    # Commit the session to save the changes to the database
    db.commit()

# Function to get statistics about the comparisons and votes
def get_statistics(db: Session) -> StatisticsResponse:
    # Get the total number of comparisons
    total_comparisons = db.query(Comparison).count()
    # Get the total number of votes for the base model
    total_votes_model_base = db.query(Comparison).with_entities(
        func.sum(Comparison.votes_model_base)
    ).scalar() or 0
    # Get the total number of votes for the lora model
    total_votes_model_lora = db.query(Comparison).with_entities(
        func.sum(Comparison.votes_model_lora)
    ).scalar() or 0

    # Return a response object with the statistics
    return StatisticsResponse(
        total_comparisons=total_comparisons,
        total_votes_model_1=total_votes_model_base,
        total_votes_model_2=total_votes_model_lora,
    )


