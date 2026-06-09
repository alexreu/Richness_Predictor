from pydantic import BaseModel


class PredictionRequest(BaseModel):
    age: int
    workclass: str
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int


class PredictionResponse(BaseModel):
    status: str
    prediction: str
