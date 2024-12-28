# Intel Image Classification with Transfer Learning

## Overview
This project implements image classification using transfer learning on the **Intel Image Dataset**. The dataset contains 25,000 images across six categories: Buildings, Forest, Glacier, Mountain, Sea, and Street. 

The task involves training a model using transfer learning with **ResNet** or **EfficientNet**, employing PyTorch Lightning and **Optuna** for hyperparameter optimization, and logging the results using **Weights & Biases (WANDB)**.

## Key Features
- **Transfer Learning**: Models trained using ResNet and EfficientNet architectures.
- **PyTorch Lightning**: Modular and scalable training pipeline with `LightningModule` and `Trainer`.
- **Hyperparameter Optimization**: Tuning backbone and learning rate with Optuna.
- **WANDB Integration**: Logs training, validation, and testing metrics for better tracking.
- **Dataset**:
  - Training: 14k images
  - Testing: 3k images
  - Prediction: 7k images

## Technologies Used
- Python
- PyTorch Lightning
- WANDB (Weights & Biases)
- Optuna
- ResNet and EfficientNet architectures

## Dataset
The dataset can be downloaded from Kaggle: [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

### Dataset Structure
- **Classes**: {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
- **Image Size**: 150x150
- **Split**:
  - Train: 14,000 images
  - Test: 3,000 images
  - Prediction: 7,000 images


## Results
- **Best Model**: [EfficientNet or ResNet]
- **Hyperparameters**:
  - Backbone: EfficientNet
  - Learning Rate: 0.001 (optimized with Optuna)
- **Performance Metrics**:
  - Training Accuracy: 90%
  - Testing Accuracy: 88%
  - Prediction Accuracy: 87%
  
## File Descriptions
- **`intel_image_classification.ipynb`**: Jupyter notebook for EDA, model training, and performance analysis.
## Instructions to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Intel-Image-Classification.git
   cd Intel-Image-Classification
