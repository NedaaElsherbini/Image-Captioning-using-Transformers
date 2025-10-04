# Image Captioning with Transformers

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)

## Project Overview
This project implements an image captioning model using a Transformer architecture combined with a pre-trained EfficientNetB0 CNN for feature extraction. Trained on the Flickr30k dataset, the model generates descriptive captions for images. A Streamlit web app allows users to upload images and view generated captions in a user-friendly interface.

## Features
- **Transformer-based Model**: Combines EfficientNetB0 for image feature extraction with a Transformer encoder-decoder for caption generation.
- **Streamlit Deployment**: A web app for uploading JPG images and generating captions.
- **Custom Text Processing**: Captions are tokenized with custom standardization, including `<start>` and `<end>` tokens.
- **Scalable Architecture**: Supports a vocabulary size of 13,000 tokens and a sequence length of 24 tokens.

## Dataset
The model is trained on the **Flickr30k dataset**, which includes 31,000 images, each paired with five captions. Pre-processing involves:
- Caption normalization and length filtering (4 to 24 tokens).
- Image resizing to 299x299 pixels.
- Dataset splitting into training (20,915 samples), validation (5,124 samples), and test (105 samples) sets.

## Usage
### Running the Streamlit App
1. Run the Streamlit app:
   ```bash
   streamlit run Image_captioning_deployment.py
   ```
2. Open the provided URL (e.g., `http://localhost:8501`) in your browser.
3. Upload a JPG image to generate a caption.


## Model Architecture
The model integrates computer vision and natural language processing through a hybrid architecture:
- **Encoder**:
  - **Feature Extraction**: Utilizes EfficientNetB0, a pre-trained CNN, to extract spatial and contextual features from input images resized to 299x299 pixels.
  - **Processing**: Features are normalized to stabilize training and passed through dense layers to project them into a 512-dimensional embedding space, aligning with the Transformer's input requirements.
  - **Attention**: Multi-head attention mechanisms process the embedded features, capturing relationships between different parts of the image.
  - **Normalization**: Layer normalization ensures stable training dynamics.
- **Decoder**:
  - **Input Processing**: Captions are tokenized (vocabulary size: 13,000) and embedded with positional encodings to preserve word order.
  - **Attention Mechanisms**: Employs masked multi-head attention to prevent attending to future tokens and cross-attention to integrate encoder outputs, enabling the model to focus on relevant image features.
  - **Feed-Forward Layers**: Dense layers with 512 units (FF_DIM) process attention outputs, followed by layer normalization for stability.
- **Output**: A final Dense layer with softmax activation generates probabilities over the 13,000-token vocabulary, producing the next token in the sequence.
- **Training**: Uses Sparse Categorical Crossentropy loss to optimize weights, with a batch size of 512 over 30 epochs. Inference employs a Greedy algorithm to generate captions token-by-token, stopping at the `<end>` token.
- **Inference**: Greedy algorithm with BLEU score evaluation.

This architecture balances computational efficiency with high-quality caption generation, leveraging EfficientNetB0's pre-trained weights and the Transformer's attention mechanisms.

## Results
The model generates coherent captions for images in the test set, with performance evaluated using BLEU scores. Example output:
![Web App Screenshot](https://github.com/NedaaElsherbini/Image-Captioning-using-Transformers/blob/main/Sample-results.png)

## Usage
### Running the Streamlit App
1. Ensure the `model.keras` file is in the project directory.
2. Launch the Streamlit app:
   ```bash
   streamlit run Image_captioning_deployment.py
   ```
3. Open the provided URL (typically `http://localhost:8501`) in a web browser.
4. Upload a JPG or JPEG image using the file uploader.
5. View the generated caption displayed below the image.

