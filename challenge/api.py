from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from challenge.model import DelayModel

app = FastAPI()

# Initialize the model
model = DelayModel()

# Dummy data to train the model during initialization
dummy_data = pd.DataFrame({
    "Fecha-I": ["2023-01-01 10:00:00", "2023-01-01 15:00:00"],
    "Fecha-O": ["2023-01-01 10:10:00", "2023-01-01 15:20:00"],
    "OPERA": ["Grupo LATAM", "Sky Airline"],
    "TIPOVUELO": ["I", "N"],
    "MES": [1, 1],
    "delay": [0, 1]
})

# Preprocess and train the model
features, target = model.preprocess(dummy_data, target_column="delay")
model.fit(features, target)

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class FlightRequest(BaseModel):
    flights: list[FlightData]

@app.get("/health")
def health_check():
    """
    Check API health.

    Returns:
        dict: Status of the API.
    """
    return {"status": "OK"}

@app.post("/predict")
def predict(request: FlightRequest):
    """
    Predict flight delays.

    Args:
        request (FlightRequest): The flight data for prediction.

    Returns:
        dict: Predictions of delays.
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([flight.dict() for flight in request.flights])

        # Validate input values
        if not input_data['MES'].between(1, 12).all():
            raise HTTPException(status_code=400, detail="Invalid value in MES column. Expected 1-12.")

        # Add dummy columns for preprocessing
        input_data = pd.get_dummies(input_data, columns=["OPERA", "TIPOVUELO"], drop_first=True)

        # Define required features
        required_features = [
            'OPERA_Latin American Wings', 'MES_7', 'MES_10', 'OPERA_Grupo LATAM',
            'MES_12', 'TIPOVUELO_I', 'MES_4', 'MES_11', 'OPERA_Sky Airline', 'OPERA_Copa Air'
        ]

        # Add missing features and ensure no duplicates
        for feature in required_features:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data.loc[:, ~input_data.columns.duplicated()]  # Remove duplicate columns

        # Reindex to match the required feature set
        input_data = input_data.reindex(columns=required_features, fill_value=0)

        # Predict delays
        predictions = model.predict(input_data)

        return {"predict": predictions}  # Adjust response format to match test expectations
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
