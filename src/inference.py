import os
import sys
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MultiLabelModel


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def load_model(model_path, num_classes=4, device='cpu'):
    model = MultiLabelModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(image_path, model, transform, device, threshold=0.5, attribute_names=None):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
    
    probs = probabilities.cpu().numpy()[0]
    preds = (probs > threshold).astype(int)
    
    if attribute_names is None:
        attribute_names = [f'Attribute_{i+1}' for i in range(len(preds))]
    
    return {
        'probabilities': probs,
        'predictions': preds,
        'present_attributes': [attribute_names[i] for i in range(len(preds)) if preds[i] == 1]
    }


def main():
    MODEL_PATH = Path(__file__).parent.parent / 'model.pth'
    IMAGES_DIR = Path(__file__).parent.parent / 'images'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found")
        print("Run train.py first")
        sys.exit(1)
    
    print("Loading model...")
    model = load_model(MODEL_PATH, device=device)
    print("Ready!\n")
    
    transform = get_transforms()
    attribute_names = ['Attr1', 'Attr2', 'Attr3', 'Attr4']
    
    while True:
        user_input = input("Enter image filename (or 'quit' to exit): ").strip()
        
        if user_input.lower() == 'quit':
            print("Bye!")
            break
        
        image_path = IMAGES_DIR / user_input
        
        if not image_path.exists():
            print(f"Not found: {image_path}\n")
            continue
        
        results = predict(image_path, model, transform, device, threshold=0.5, attribute_names=attribute_names)
        
        print(f"\nResults for '{user_input}':")
        print(f"  Probs: {results['probabilities']}")
        print(f"  Preds: {results['predictions']}")
        
        if results['present_attributes']:
            print(f"  Attrs: {', '.join(results['present_attributes'])}")
        else:
            print(f"  Attrs: None")
        print()


if __name__ == '__main__':
    main()
