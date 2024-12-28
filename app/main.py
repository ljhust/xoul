from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base
from app import crud, schemas, mock_model
from app.query import get_response

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application
app = FastAPI(title="Comparison API", version="1.0.0")

# Dependency to get the database session
def get_db():
    db = SessionLocal()  # Create a new database session
    try:
        yield db  # Provide the session to the caller
    finally:
        db.close()  # Ensure the session is closed after use

@app.post("/compare", response_model=schemas.CompareResponse)
async def compare(input_text: schemas.CompareRequest, db: Session = Depends(get_db)):
    """
    Input text and receive responses from two models.
    """
    try:
        # mock data to test basic function
        # response_model_1 = mock_model.get_response(input_text.text, model="model_1")
        # response_model_2 = mock_model.get_response(input_text.text, model="model_2")
        # real call to model
        response_model_base = get_response(input_text.text, "sophosympatheia/Midnight-Miqu-70B-v1.0")
        response_model_lora = get_response(input_text.text, "sophosympatheia/Midnight-Miqu-70B-v1.0-loar")
        
        # Create a comparison entry in the database
        comparison_data = crud.create_comparison(
            db, input_text.text, response_model_base, response_model_lora
        )
        return comparison_data  # Return the comparison data
    except Exception as e:
        # Raise an HTTP exception if an error occurs
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vote/{comparison_id}/{model_choice}")
async def vote(
    comparison_id: int, model_choice: str, db: Session = Depends(get_db)
):
    """
    Vote for the preferred model response.
    """
    # Validate the model choice
    if model_choice not in ["model_base", "model_lora"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model choice. Must be 'model_base' or 'model_lora'.",
        )

    try:
        # Record the vote in the database
        crud.cast_vote(db, comparison_id, model_choice)
        return {"message": "Vote recorded successfully."}  # Return success message
    except Exception as e:
        # Raise an HTTP exception if an error occurs
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics", response_model=schemas.StatisticsResponse)
async def statistics(db: Session = Depends(get_db)):
    """
    View basic statistics about evaluator preferences.
    """
    try:
        # Retrieve statistics from the database
        stats = crud.get_statistics(db)
        return stats  # Return the statistics
    except Exception as e:
        # Raise an HTTP exception if an error occurs
        raise HTTPException(status_code=500, detail=str(e))
