import csv
import glob
import json
import torch
from pathlib import Path
from model import Resnet
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

dev = torch.device('cpu')

def get_num_workers():
    suggested_workers = 0
    if hasattr(os, 'sched_getaffinity'):
        try:
            suggested_workers = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if suggested_workers == 0:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            suggested_workers = cpu_count
    num_workers = min(8, suggested_workers)
    return num_workers

test_dataset = datasets.ImageFolder('dataset/test')
loader_test = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=get_num_workers())


model=Resnet(num_classes = len(test_dataset.classes)).to(dev)
state_dict=torch.load('training/model/best_model.pt', map_location=dev)
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 3, 64, 130)
torch.onnx.export(
	model,
	dummy_input,
	'training/model/best_model.onnx',
	export_params=True,
	opset_version=14,
	do_constant_folding=True,
	input_names=['input'],
	output_names=['output'],
	dynamo=False,
)
print('Success!')

# Collect every callout value from the per-map conversion files into one
# alphabetically sorted CSV, one unique callout per line.
callouts = set()
for path in glob.glob('dataset_transforms/callout_conversion/*.json'):
	with open(path, 'r') as f:
		for value in json.load(f).keys():
			callouts.add(value)

with open('training/model/callouts.csv', 'w', newline='') as f:
	writer = csv.writer(f)
	for callout in sorted(callouts, key=str.lower):
		writer.writerow([callout])
	writer.writerow(["None"])

print(f'Wrote {(len(callouts) + 1)} callouts to training/model/callouts.csv')
