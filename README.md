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

## Installation
To set up the project environment:

`git clone https://github.com/boemer00/signet-ring-cell-classification.git`

`cd signet-ring-cell-classification`

`pip install -r requirements.txt`

## Usage
To train the model, run:

`python src/trainer.py`

## Running Tests
To run tests for this project, navigate to the project directory and activate a virtual environment (recommended). Use the pytest command to execute tests. For detailed output, use `pytest -v`. If using *pytest-cov* for test coverage, view the report with `pytest --cov=src`

## Model Architecture and Training

The classification model is built on the VGG19 architecture, a deep convolutional network known for its efficacy in image recognition tasks. This project uses VGG19 as a base model and it includes the following customisations:

- **Base Model**: VGG19 pre-trained on ImageNet, without the top layer, to leverage pre-existing feature maps.
- **Global Average Pooling**: to reduce dimensionality and summarise features globally.
- **Dense Layer**: a fully connected layer with 512 neurons and ReLU activation, including an L2 regulariser for weight decay to reduce overfitting.
- **Dropout**: to further prevent overfitting, a dropout rate of 0.5 is applied.
- **Output Layer**: a single neuron with a sigmoid activation function to output the probability of the presence of signet ring cells.

The model is compiled with the Adam optimiser and a learning rate of 0.001, using binary cross-entropy as the loss function. It includes accuracy and recall metrics to track performance during training.

Training is conducted over 50 epochs with real-time data augmentation and validation using respective data generators. The model file is inside `src/trainer.py`.

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
