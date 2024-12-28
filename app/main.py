from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database import engine, SessionLocal, Base
from app import crud, schemas, mock_model
import uvicorn

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Comparison API", version="1.0.0")


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/compare", response_model=schemas.CompareResponse)
async def compare(input_text: schemas.CompareRequest, db: Session = Depends(get_db)):
    """
    Input text and receive responses from two models.
    """
    try:
        response_model_1 = mock_model.get_response(input_text.text, model="model_1")
        response_model_2 = mock_model.get_response(input_text.text, model="model_2")
        comparison_data = crud.create_comparison(
            db, input_text.text, response_model_1, response_model_2
        )
        return comparison_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vote/{comparison_id}/{model_choice}")
async def vote(
    comparison_id: int, model_choice: str, db: Session = Depends(get_db)
):
    """
    Vote for the preferred model response.
    """
    if model_choice not in ["model_1", "model_2"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid model choice. Must be 'model_1' or 'model_2'.",
        )

    try:
        crud.cast_vote(db, comparison_id, model_choice)
        return {"message": "Vote recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics", response_model=schemas.StatisticsResponse)
async def statistics(db: Session = Depends(get_db)):
    """
    View basic statistics about evaluator preferences.
    """
    try:
        stats = crud.get_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)