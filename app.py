from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
import joblib 
import numpy as np 
import gdown
import os

app = FastAPI(    
    title="ML Model API",    
    description="Machine Learning Model deployed using FastAPI",    
    version="1.0.0" 
    )

FILE_ID = "1VuWu4AhEzkrHVySdHXaAQK45tqJPRC8a"
MODEL_PATH = "model.pkl"

# Download model only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print("Model not found. Downloading...")

    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_PATH,
        quiet=False
    )

else:
    print("Model already exists. Skipping download.")

# Load model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")

# MODEL_PATH = "model.pkl" 
# try:    
#     model = joblib.load(MODEL_PATH)    
#     print("Model loaded successfully") 
# except Exception as e:    
#     print("Error loading model:", e)    
#     model = None

class RegressionInput(BaseModel):    
    holiday: int
    rain_1h: float 
    snow_1h: float
    clouds_all: int 
    week: int 
    hour: int
    temp_celsius: float 
    year: int
    month: int 
    day: int 
    weather_main_Clear: float
    weather_main_Clouds: float 
    weather_main_Drizzle: float 
    weather_main_Fog: float
    weather_main_Haze: float
    weather_main_Mist: float 
    weather_main_Rain: float
    weather_main_Smoke: float 
    weather_main_Snow: float 
    weather_main_Squall: float
    weather_main_Thunderstorm: float 
    session_Afternoon: float 
    session_Evening: float
    session_Morning: float 
    session_Night: float

@app.get("/") 
def home():    
    return {        
        "message": "ML Model API is running",        
        "status": "success"    
    }

@app.get("/health") 
def health():    
    if model is None:        
        return {"status": "unhealthy", "model_loaded": False}    
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict") 
def predict_price(data: RegressionInput):    
    input_data = np.array([[        
        data.holiday, data.rain_1h, data.snow_1h, data.clouds_all, data.week, data.hour,
data.temp_celsius, data.year, data.month, data.day, data.weather_main_Clear,
data.weather_main_Clouds, data.weather_main_Drizzle, data.weather_main_Fog,
data.weather_main_Haze, data.weather_main_Mist, data.weather_main_Rain,
data.weather_main_Smoke, data.weather_main_Snow, data.weather_main_Squall,
data.weather_main_Thunderstorm, data.session_Afternoon, data.session_Evening,
data.session_Morning, data.session_Night    
        ]])    
    prediction = model.predict(input_data)[0]    
    return {        
        "prediction": round(prediction),        
        "unit": "Vehicles"    
    }