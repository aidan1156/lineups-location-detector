import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
#other
import copy
import json
import os

from model import Resnet

torch.manual_seed(67)

transform = transforms.Compose(
        [
            transforms.RandomAffine(
                translate=(0.1, 0.1),
                scale=(1, 1.2),
                degrees=(-0.1, 0.1),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Resize(64),
            transforms.CenterCrop((130, 64)),
            transforms.ToTensor(),
        ]
    )

train_path = 'text-dataset/train'
test_path = 'text-dataset/test'
val_path = 'text-dataset/val'

train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

train_targets = torch.tensor(train_dataset.targets)
# Get class counts from these specific targets
# Use torch.unique to count occurrences of each class in the training split
classes, class_counts = torch.unique(train_targets, return_counts=True)
class_weights = 1. / class_counts.float()
# Map the weights to every sample in the training subset
sample_weights = class_weights[train_targets]

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
print(len(train_dataset), len(val_dataset), len(test_dataset))

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

def check_accuracy(loader: DataLoader, model: Resnet):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return float(num_correct) / float(num_samples) * 100

max_epochs = 100
batch_size = 128
num_workers = get_num_workers()


loader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
dtype = torch.float32

num_classes = len(train_dataset.classes)

model = Resnet(num_classes)
optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

best_accuracy = 0
best_accuracy_epoch = -1
best_model_state = None

model = model.to(device=device)  # move the model parameters to CPU/GPU
for e in range(max_epochs):
    # if no improvement in 10 consecutive epochs stop training
    if best_accuracy_epoch + 10 <= e:
        print(f'Early stopping at epoch {e} with best accuracy {best_accuracy:.2f} at epoch {best_accuracy_epoch}')
        model.load_state_dict(best_model_state)
        break
    for t, (x, y) in enumerate(loader_train):
        model.train()  # put model to training mode
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)

        scores = model(x)
        loss = F.cross_entropy(scores, y)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        loss.backward()

        # Update the parameters of the model using the gradients
        optimizer.step()

        if t % 1000 == 0:
            print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
            
    scheduler.step()
    print("LR now is ", optimizer.param_groups[0]["lr"])
    
    accuracy = check_accuracy(loader_val, model)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_accuracy_epoch = e
        best_model_state = copy.deepcopy(model.state_dict())

    if not os.path.exists('training'):
        os.makedirs('training')
    torch.save(model.state_dict(), 'training/current_model.pt')
    torch.save(best_model_state, 'training/best_model.pt')
    metadata = {
        'best_accuracy': best_accuracy,
        'best_accuracy_epoch': best_accuracy_epoch,
        'last_epoch': e
    }
    with open('training/training_metadata.json', 'w') as f:
        json.dump(metadata, f)