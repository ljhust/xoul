from sqlalchemy import Column, Integer, String, Text
from app.database import Base

# Define a class 'Comparison' that inherits from 'Base'
class Comparison(Base):
    # Specify the name of the table in the database
    __tablename__ = "comparisons"

    # Define the columns of the table
    id = Column(Integer, primary_key=True, index=True)  # Primary key column with an index
    input_text = Column(Text, nullable=False)  # Column for input text, cannot be null
    response_model_base = Column(Text, nullable=False)  # Column for base model response, cannot be null
    response_model_lora = Column(Text, nullable=False)  # Column for LoRA model response, cannot be null
    votes_model_base = Column(Integer, default=0)  # Column for votes for the base model, default is 0
    votes_model_lora = Column(Integer, default=0)  # Column for votes for the LoRA model, default is 0
