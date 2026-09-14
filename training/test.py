from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import Resnet


def get_num_workers():
	suggested_workers = 0
	if hasattr(os, "sched_getaffinity"):
		try:
			suggested_workers = len(os.sched_getaffinity(0))
		except Exception:
			pass
	if suggested_workers == 0:
		cpu_count = os.cpu_count()
		if cpu_count is not None:
			suggested_workers = cpu_count
	return min(8, suggested_workers)


def load_state_dict(model: Resnet, checkpoint_path: Path, device: torch.device):
	checkpoint = torch.load(checkpoint_path, map_location=device)
	if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint
	model.load_state_dict(state_dict)
	return model


def evaluate(model: Resnet, loader: DataLoader, device: torch.device):
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device=device, dtype=torch.float32)
			labels = labels.to(device=device, dtype=torch.long)

			logits = model(images)
			predictions = torch.argmax(logits, dim=1)
			correct += (predictions == labels).sum().item()
			total += labels.size(0)

	return correct, total


def main():
	checkpoint_path = Path("training/model/best_model.pt")
	test_path = Path("dataset/test")

	if not checkpoint_path.exists():
		raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
	if not test_path.exists():
		raise FileNotFoundError(f"Test dataset not found: {test_path}")

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# Must match the shape train.py feeds the model; the random affine jitter it
	# uses is training-only, so evaluation just resizes and crops.
	transform = transforms.Compose(
		[
			transforms.Resize(64),
			transforms.CenterCrop((64, 130)),
			transforms.ToTensor(),
		]
	)

	test_dataset = datasets.ImageFolder(test_path, transform=transform)
	test_loader = DataLoader(
		test_dataset,
		batch_size=128,
		shuffle=False,
		num_workers=get_num_workers(),
	)

	model = Resnet(num_classes=len(test_dataset.classes)).to(device)
	model = load_state_dict(model, checkpoint_path, device)

	correct, total = evaluate(model, test_loader, device)
	accuracy = 100.0 * correct / total if total else 0.0

	print(f"Loaded checkpoint: {checkpoint_path}")
	print(f"Test samples: {total}")
	print(f"Correct predictions: {correct}")
	print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
	main()