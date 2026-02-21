import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from inference import load_model, predict, get_transforms

device = 'cpu'
model_path = project_root / 'model.pth'
images_dir = project_root / 'images'

model = load_model(str(model_path), device=device)
transform = get_transforms()

# Test on a few images
test_images = ['image_0.jpg', 'image_2.jpg', 'image_15.jpg', 'image_100.jpg']
attribute_names = ['Attr1', 'Attr2', 'Attr3', 'Attr4']

print("=" * 60)
print("INFERENCE TEST RESULTS")
print("=" * 60)

for img_name in test_images:
    img_path = images_dir / img_name
    if img_path.exists():
        results = predict(str(img_path), model, transform, device, threshold=0.5, attribute_names=attribute_names)
        attrs = ', '.join(results['present_attributes']) if results['present_attributes'] else 'None'
        probs = ', '.join([f'{p:.3f}' for p in results['probabilities']])
        print(f"\n{img_name}:")
        print(f"  Probabilities: [{probs}]")
        print(f"  Predictions:   {list(results['predictions'])}")
        print(f"  Present:       {attrs}")

print("\n" + "=" * 60)
