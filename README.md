# Signet Ring Cell Classification Model

## Introduction
This project presents a classification algorithm of signet ring cell carcinoma in histopathological images. Ultimately, the projects addresses the significant challenge of early detection of rare but aggressive carcinomas.

## Background
Signet ring cell carcinoma is a rare form of adenocarcinoma, known for its poor prognosis. So, accurate detection of these cells is key to improving patient survival rates. The project uses a unique dataset comprising both positive and negative samples to train a classification model.

## Dataset Description
The dataset includes:
- **Positive Samples**: 77 images from 20 Whole Slide Images (WSIs).
- **Negative Samples**: 378 images from 79 WSIs, potentially containing other tumor cells but no signet ring cells.

Each image is 2000x2000 pixels, sourced from gastric mucosa and intestine tissues, stained with hematoxylin and eosin, and scanned at X40 magnification. While the original dataset includes annotated information, this project does not require such information.

[.](docs/signet_ring_computer_vision_model.png)

## Installation
To set up the project environment:

`git clone https://github.com/boemer00/signet-ring-cell-classification.git`

`cd signet-ring-cell-classification`

`pip install -r requirements.txt`

## Usage

### Training the Model

To train the model, execute the following command:

`python src/trainer.py`

This script uses the model in `src/trainer.py` and saves the trained model for future use.

### Making Predictions Using FastAPI

This project also includes a FastAPI application for deploying the trained model as a web service, which allows users to upload images and receive predictions.

#### Starting the FastAPI Server

To start the FastAPI server, run:

`uvicorn app.main:app --reload`

This command will launch the server, making the API accessible at `http://127.0.0.1:8000`. The `--reload` flag is useful during development as it automatically reloads the server when code changes are made.

#### Using the Prediction Endpoint

Once the server is running, you can make predictions by sending image files to the `/predict/` endpoint. This can be done using the Swagger UI:

1. Navigate to `http://127.0.0.1:8000/docs` in your web browser.
2. Locate the `/predict/` endpoint and use the interactive UI to upload an image file.
3. Submit the request, and the API will return the model's prediction for the uploaded image.

The FastAPI application uses the `SignetRingPredictor` class from `src/predictor.py` for processing the image and generating predictions.


## Running Tests
To run tests for this project, navigate to the project directory and activate a virtual environment (recommended). Use the pytest command to execute tests. For detailed output, use `pytest -v`. If using *pytest-cov* for test coverage, view the report with `pytest --cov=src`

## Model Architecture and Training

The classification model is built on the VGG19 architecture, with specific customizations for signet ring cell classification. The model includes a `SignetRingPredictor` class, which encapsulates the model's loading and prediction logic, making it seamlessly integrable with FastAPI for deployment.

- **Base Model**: VGG19 pre-trained on ImageNet, without the top layer.
- **Custom Layers**: Global Average Pooling, Dense, Dropout, and Output layers tailored for binary classification.
- **Compilation**: Compiled with the Adam optimizer, binary cross-entropy loss, and includes accuracy and recall metrics.

The model's architecture and training details are elaborated in `src/trainer.py`. The `SignetRingPredictor` class, which you can use for predictions--for more details see `src/predictor.py`.

## Results and Evaluation

The evaluation of the classification model focuses on the same three metrics established in the paper:

- **Recall**: The model is designed to prioritise recall, ensuring that the rate of missed signet ring cells (false negatives) is above the minimum acceptable threshold of 20%. Recall is crucial for medical applications where the cost of missing positive cases is high.

- **Normal Region FP (FPNormal)**: This metric measures the average number of false positives per image in normal (non-cancerous) regions, providing insight into the specificity of the model. A lower FPNormal value indicates a model with better specificity, reducing the likelihood of false alarms in clinical practice.

- **Free-response ROC (FROC)**: The FROC curve is a plot of the true positive rate against the average number of false positives per image. It is a variantion of the well-established ROC curve. This metric shows the model's ability to maintain a balance between sensitivity (*aka* recall) and the false positive rate across different thresholds.

## Citation
Da Q, Huang X, Li Z, et al. [DigestPath: a Benchmark Dataset with Challenge Review for the
Pathological Detection and Segmentation of Digestive-System](https://doi.org/10.1016/j.media.2022.102485)[J].
Medical Image Analysis, 2022: 102485.

[GitHub repo for the original project](https://github.com/bupt-ai-cz/CAC-UNet-DigestPath2019/blob/main/papers/DigestPath-a-Benchmark-Dataset-with-Challenge-Review.pdf)
