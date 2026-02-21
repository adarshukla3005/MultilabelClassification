import torch
import numpy as np
from pathlib import Path


def parse_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                image_name = parts[0]
                attrs = []
                for i in range(1, 5):
                    if parts[i].upper() == 'NA':
                        attrs.append(-1)
                    else:
                        attrs.append(int(parts[i]))
                labels[image_name] = attrs
    return labels


def calculate_class_weights(labels_dict, num_classes=4):
    class_counts = np.zeros(num_classes)
    
    for attrs in labels_dict.values():
        for i, attr in enumerate(attrs):
            if attr == 1:
                class_counts[i] += 1
    
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = 1.0 / class_counts[i]
        else:
            weights[i] = 1.0
    
    weights = weights / weights.sum() * num_classes
    return weights


def create_mask(labels, num_classes=4):
    mask = torch.ones(num_classes, dtype=torch.float32)
    for i, label in enumerate(labels):
        if label == -1:
            mask[i] = 0.0
    return mask


def get_available_images(labels_dict, images_dir):
    images_dir = Path(images_dir)
    available_images = []
    for image_name in labels_dict.keys():
        if (images_dir / image_name).exists():
            available_images.append(image_name)
    return available_images
