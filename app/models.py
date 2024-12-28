from sqlalchemy import Column, Integer, String, Text
from app.database import Base


class Comparison(Base):
    __tablename__ = "comparisons"

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(Text, nullable=False)
    response_model_1 = Column(Text, nullable=False)
    response_model_2 = Column(Text, nullable=False)
    votes_model_1 = Column(Integer, default=0)
    votes_model_2 = Column(Integer, default=0)
