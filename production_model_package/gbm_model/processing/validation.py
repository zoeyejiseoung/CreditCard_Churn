from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from gbm_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleCreditCardDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class CreditCardDataInputSchema(BaseModel):
    Customer_Age: Optional[int]
    Dependent_count: Optional[int]
    Months_on_book: Optional[int]
    Total_Relationship_Count: Optional[int]
    Months_Inactive_12_mon: Optional[int]
    Contacts_Count_12_mon: Optional[int]
    Credit_Limit: Optional[float]
    Total_Revolving_Bal: Optional[float]
    Avg_Open_To_Buy: Optional[float]
    Total_Amt_Chng_Q4_Q1: Optional[float]
    Total_Trans_Amt: Optional[int]
    Total_Trans_Ct: Optional[int]
    Total_Ct_Chng_Q4_Q1: Optional[float]
    Avg_Utilization_Ratio: Optional[float]
    Gender: Optional[str]
    Education_Level: Optional[str]
    Marital_Status: Optional[str]
    Income_Category: Optional[str]
    Card_Category: Optional[str]


class MultipleCreditCardDataInputs(BaseModel):
    inputs: List[CreditCardDataInputSchema]
