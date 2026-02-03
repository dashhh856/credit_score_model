
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load("credit+approval/model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class CreditApplication(BaseModel):
    A1: str
    A2: float
    A3: float
    A4: str
    A5: str
    A6: str
    A7: str
    A8: float
    A9: str
    A10: str
    A11: int
    A12: str
    A13: str
    A14: float
    A15: float

@app.post("/predict")
def predict(data: CreditApplication):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert input to DataFrame
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    
    # Ensure correct data types if necessary (though pydantic helps)
    # The model pipeline handles preprocessing
    
    try:
        prediction = model.predict(df)[0]
        # prediction is likely '+' or '-' based on the dataset
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
