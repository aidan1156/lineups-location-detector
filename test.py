# Load the best model and run inference on image paths entered in the terminal.
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import Resnet
from dataset import LineupDataset, EvalTransform

def load_model(map_name, model_path=None, device=None):
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	dataset = LineupDataset(map_name=map_name, transform=None)
	model = Resnet(num_classes=dataset.get_num_classes()).to(device)

	if model_path is None:
		model_path = Path(f"models/best_model_{map_name}.pth")
	else:
		model_path = Path(model_path)

	if not model_path.exists():
		raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

	state_dict = torch.load(model_path, map_location=device)
	model.load_state_dict(state_dict)
	model.eval()
	return model, dataset, device


def predict_image(model, image_path, transform, idx_to_label, device):
	image = Image.open(image_path).convert("RGB")
	tensor = transform(image).unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(tensor)
		probs = torch.softmax(logits, dim=1)
		confidence, pred_idx = torch.max(probs, dim=1)

	pred_idx = int(pred_idx.item())
	confidence = float(confidence.item())
	pred_label = idx_to_label[pred_idx]
	return pred_label, confidence


def main():
	map_name = input("Map name (default Ascent): ").strip() or "Ascent"
	model_path_input = input("Model path (press Enter for default): ").strip()
	model_path = model_path_input if model_path_input else None

	model, dataset, device = load_model(map_name=map_name, model_path=model_path)
	transform = EvalTransform(output_size=(384, 216))

	print(f"Loaded model for map '{map_name}' on {device}.")
	print("Enter image paths one-by-one. Type 'q' to quit.")

	while True:
		raw = input("Image path: ").strip()
		if not raw:
			continue
		if raw.lower() in {"q", "quit", "exit"}:
			break

		image_path = Path(raw)
		if not image_path.exists():
			print(f"File not found: {image_path}")
			continue

		try:
			label, confidence = predict_image(
				model=model,
				image_path=image_path,
				transform=transform,
				idx_to_label=dataset.idx_to_label,
				device=device,
			)
			print(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
		except Exception as exc:
			print(f"Could not process '{image_path}': {exc}")


if __name__ == "__main__":
	main()

