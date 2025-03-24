# 🔍 OCR From Scratch in PyTorch

A character-level Optical Character Recognition (OCR) system built from scratch using PyTorch.

This project demonstrates how to train a neural network to recognize characters from images, with a modular training loop and dataset preprocessing pipeline — perfect for research, education, or lightweight OCR deployment.

---

## 🚀 What It Does

- Loads and processes labeled character images  
- Trains a CNN-based classifier from scratch  
- Predicts character outputs from unseen test images  
- Optionally supports evaluation and inference via CLI or script  

---

## 🛠️ Tech Stack

| Component     | Technology         |
|--------------|--------------------|
| Model        | PyTorch (CNN)      |
| Data Format  | Custom image + label pairs |
| Training     | PyTorch Dataloaders + Adam |
| Evaluation   | Accuracy, Confusion Matrix |
| Inference    | Command-line interface |


---

## 🧠 Model Architecture

- 3 Convolutional layers + BatchNorm + ReLU
- Fully connected output layer for character classification
- Softmax activation for predictions

Supports alphanumeric or customizable character sets.

---

## 🧪 Training Example

```bash
python train.py \
    --data-dir ./data/train \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001
```

## Inference Example
```
python predict.py \
    --image ./data/test/sample_A.png \
    --weights ./models/ocr_cnn.pth
```

## Metrics
- Accuracy
- Per-character precision/recall (optional)
- Confusion matrix (via evaluate.py)

## Acknowledgments
Built from scratch using PyTorch, inspired by open research in OCR systems.

## Future Work
Add Transformer-based OCR architecture
Multi-line text detection and decoding
Export trained model to ONNX or TorchScript for deployment

## 📫 Contact
Author: Jordan Shamai
Email: jordan.shamai04@gmail.com
Project Type: Research Prototype
