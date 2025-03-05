# MobileNetV2-Optimization-for-Embedded-Computer-Vision
## Overview
This project demonstrates how to optimize a MobileNetV2 model for deployment on resource-constrained embedded systems. By applying transfer learning, fine-tuning, and FP16 quantization techniques, the model achieves high accuracy on the CIFAR-10 dataset while significantly reducing its size and memory footprint.

## Key Features
Two-phase training approach with transfer learning from ImageNet weights
Mixed precision training with FP16 to reduce memory consumption
Data augmentation pipeline to improve model generalization
Model quantization with TensorFlow Lite for embedded deployment
76.89% reduction in model size (from 18.5MB to 4.28MB) with minimal accuracy loss (0.08%)

## Results
| Model                  | Size       | Accuracy |
|------------------------|------------|----------|
| Keras (.h5)            | 18.50 MB   | 90.13%   |
| TFLite FP16 (.tflite)  | 4.28 MB    | 90.05%   |

## Project Structure

├── mobilenetv2_optimization_embedded_fp16.py   # Main implementation script

├── transfer_learning_model.h5                  # Full precision Keras model

├── model_fp16.tflite                           # Quantized TFLite model for deployment

└── README.md                                   # This file

## Requirements 
tensorflow>=2.7.0

tensorflow-model-optimization

numpy

scikit-learn

## Implementation Details
### 1. Data Preparation

CIFAR-10 dataset with a 80/20 train-validation split
Custom preprocessing pipeline with resizing to 128x128
Data augmentation including random flips, brightness and contrast adjustments

### 2. Training Methodology

Phase 1: Train only the classification layers while keeping the base MobileNetV2 model frozen
Phase 2: Fine-tune the last 20 layers of the base model with a reduced learning rate
Training optimizations including early stopping, learning rate reduction, and dropout

### 3. Model Optimization

Mixed precision training to reduce memory footprint
TensorFlow Lite conversion with FP16 quantization
Performance validation comparing original and optimized models

## Future Work

Explore INT8 quantization for further size reduction
Implement pruning techniques to reduce model complexity
Benchmark inference time on various embedded platforms
Explore knowledge distillation to improve accuracy of compressed models

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Youssef Ennouri - March 2025
