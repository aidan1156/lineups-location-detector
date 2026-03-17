import torch
from pathlib import Path
from model import Resnet
from dataset import LineupDataset
dev=torch.device('cpu')
d=LineupDataset('Ascent', transform=None)
model=Resnet(num_classes=d.get_num_classes()).to(dev)
state_dict=torch.load('models/best_model_Ascent.pth', map_location=dev)
model.load_state_dict(state_dict)
model.eval()
dummy_input = torch.randn(1, 3, 216, 384)
torch.onnx.export(
	model,
	dummy_input,
	'Ascent_single.onnx',
	export_params=True,
	opset_version=14,
	do_constant_folding=True,
	input_names=['input'],
	output_names=['output'],
	dynamo=False,
)
print('Success!')
