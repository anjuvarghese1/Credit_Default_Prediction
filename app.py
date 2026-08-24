"""FastAPI serving layer for the credit-risk model.
Runs as a normal ASGI app locally, and as a Lambda handler in AWS
(via the Mangum adapter at the bottom).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from mangum import Mangum

MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model.joblib"

FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

model = joblib.load(MODEL_PATH)


class Applicant(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    DebtRatio: float
    MonthlyIncome: Optional[float] = None
    NumberOfOpenCreditLinesAndLoans: int
    NumberRealEstateLoansOrLines: int
    NumberOfDependents: Optional[float] = None
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    NumberOfTimes90DaysLate: int
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., alias="NumberOfTime60-89DaysPastDueNotWorse"
    )

    model_config = {"populate_by_name": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(applicant: Applicant) -> dict:
    row = {
        "RevolvingUtilizationOfUnsecuredLines": applicant.RevolvingUtilizationOfUnsecuredLines,
        "age": applicant.age,
        "DebtRatio": applicant.DebtRatio,
        "MonthlyIncome": applicant.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans": applicant.NumberOfOpenCreditLinesAndLoans,
        "NumberRealEstateLoansOrLines": applicant.NumberRealEstateLoansOrLines,
        "NumberOfDependents": applicant.NumberOfDependents,
        "NumberOfTime30-59DaysPastDueNotWorse": applicant.NumberOfTime30_59DaysPastDueNotWorse,
        "NumberOfTimes90DaysLate": applicant.NumberOfTimes90DaysLate,
        "NumberOfTime60-89DaysPastDueNotWorse": applicant.NumberOfTime60_89DaysPastDueNotWorse,
    }
    X = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    proba = float(model.predict_proba(X)[0, 1])
    return {
        "default_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
    }


# Lambda entrypoint: Mangum wraps the ASGI app so AWS Lambda can invoke it.
handler = Mangum(app)
