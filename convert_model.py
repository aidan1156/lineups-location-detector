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

test_dataset = datasets.ImageFolder('text-dataset/test')
loader_test = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=get_num_workers())


model=Resnet(num_classes = len(test_dataset.classes)).to(dev)
state_dict=torch.load('training/best_model.pt', map_location=dev)
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 3, 64, 130)
torch.onnx.export(
	model,
	dummy_input,
	'training/best_model.onnx',
	export_params=True,
	opset_version=14,
	do_constant_folding=True,
	input_names=['input'],
	output_names=['output'],
	dynamo=False,
)
print('Success!')
