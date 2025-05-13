# Deep Learning Coursework - Action Classification

This project implements a deep learning model for action classification using video frames, leveraging a CNN-LSTM architecture.

## Structure
- `cnn_lstm_model.py`: Contains the CNN-LSTM model definition.
- `Deep_Learning_Coursework.ipynb`: Jupyter notebook showcasing the training pipeline and evaluation.

## Model Details
The model uses a ResNet34 backbone to extract spatial features from frames and an LSTM to capture temporal dependencies. 

### CNN-LSTM Architecture:
1. **CNN Feature Extractor:**
   - Backbone: ResNet34
   - Pre-trained on ImageNet
   - Fine-tuned on action classification data

2. **LSTM Classifier:**
   - Hidden Dimension: 256
   - Layers: 2
   - Dropout: 0.5

### Training and Evaluation:
The notebook contains:
- Data preprocessing and augmentation
- Model training with loss and accuracy tracking
- Evaluation on validation and test sets

## Usage:
To run the notebook:
```bash
jupyter notebook Deep_Learning_Coursework.ipynb
```

To train the model using the script:
```bash
python cnn_lstm_model.py
```

## Requirements:
- torch
- torchvision
- jupyter
- numpy

To install:
```bash
pip install torch torchvision jupyter numpy
```
