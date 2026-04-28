from fastapi import FastAPI

import pandas as pd 
import uvicorn
import joblib
from fastapi import HTTPException

app = FastAPI()

model_pipe = joblib.load("model.pkl")
data = "data.csv"


@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        df = df.reindex(columns=model_pipe.feature_names_in_, fill_value=0)
        prediction = model_pipe.predict(df)
        return {"prediction": prediction.tolist()[0]}
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("app:app" , host = "0.0.0.0" , port = 8000)
    