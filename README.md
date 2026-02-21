# Multilabel Image Classification

A complete solution for training and running multilabel image classification with PyTorch.

## What's Included

- **Training code** (src/train.py) - Trains ResNet18 on 972 images with 4 binary attributes
- **Inference code** (src/inference.py) - Makes predictions on new images
- **Trained model** (model.pth) - 44.8 MB ResNet18 model
- **Loss curve** (loss_plot.png) - Training visualization
- **Dataset** (images/) - 972 training images with labels

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
cd src
python train.py
```

3. Run inference:
```bash
python inference.py
```

3. Validate everything works:
```bash
python VALIDATE.py
```

## How It Works

The project classifies images into 4 independent attributes. Each attribute is a binary classification (present or absent). The model handles a few special cases:

- **Missing labels**: Some images don't have all 4 labels. These are handled with masked loss - the model trains on available labels only.
- **Class imbalance**: Attribute 4 is rare (only 3% positive cases). The loss uses weights to prevent the model from ignoring this rare pattern.
- **Transfer learning**: Uses ResNet18 pretrained on ImageNet, fine-tuned on your data.

## Project Structure

```
src/
  train.py        - Training script
  inference.py    - Predictions
  dataset.py      - Custom Dataset
  model.py        - ResNet18 definition
  utils.py        - Helper functions
  plot_loss.py    - Plotting utilities

images/           - Training images (972)
labels.txt        - Image labels
model.pth         - Trained model
loss_plot.png     - Training loss curve
requirements.txt  - Dependencies
```

## Configuration

You can adjust training parameters in src/train.py:
```python
BATCH_SIZE = 32         # Training batch size
NUM_EPOCHS = 10         # Number of epochs
LEARNING_RATE = 1e-3    # Learning rate
```

## Troubleshooting

**Missing PyTorch**: `pip install -r requirements.txt`

**Image not found during inference**: Use filenames from labels.txt (e.g., image_0.jpg)

**Model.pth missing**: Run training first with `python src/train.py`

For more technical details, see src/README.md
