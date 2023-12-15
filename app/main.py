from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from src.predictor import SignetRingPredictor

app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_model', 'signet_ring_model.keras')

predictor = SignetRingPredictor(model_path)

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        with open('temp_image.jpeg', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        prediction = predictor.predict('temp_image.jpeg')
        os.remove('temp_image.jpeg')

        # convert np array to a standard float
        prediction_value = float(prediction[0])

        return JSONResponse(content={'prediction': prediction_value})
    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={'message': f'Error during prediction: {str(e)}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
