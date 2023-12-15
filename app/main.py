from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
from src.predictor import SignetRingPredictor

app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models', 'signet_ring_model.h5')
predictor = SignetRingPredictor(model_path)

@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # save uploaded file to disk
        with open('temp_image.jpeg', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)

        # make prediction
        prediction = predictor.predict('temp_image.jpeg')
        os.remove('temp_image.jpeg')  # remove the saved image after prediction

        return JSONResponse(content={'prediction': prediction[0]})
    except Exception as e:
        return JSONResponse(status_code=500, content={'message': f'Error during prediction: {str(e)}'})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
