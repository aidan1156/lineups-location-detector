import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model import Resnet
from fullscreen_detection.dataset import LineupDataset, TransformedSubset, EvalTransform, TrainTransform

# set seed for everything
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    # shuffle the dataset
    dataset = torch.utils.data.Subset(dataset, torch.randperm(total_size))
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])






MAP_NAME = None#'Ascent'

train_transform = TrainTransform(output_size=(135, 64))
eval_transform = EvalTransform(output_size=(135, 64))

full_dataset = LineupDataset(MAP_NAME, transform=None)
train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)

train_dataset = TransformedSubset(train_dataset, transform=train_transform)
val_dataset = TransformedSubset(val_dataset, transform=eval_transform)
test_dataset = TransformedSubset(test_dataset, transform=eval_transform)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Resnet(num_classes=full_dataset.get_num_classes()).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_model = None
best_model_accuracy = 0
best_model_accuracy_at_epoch = 0
patience = 5

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0.0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()
    
    if accuracy >= best_model_accuracy:
        best_model = model.state_dict()
        best_model_accuracy = accuracy
        best_model_accuracy_at_epoch = epoch

        # save the best model
        torch.save(best_model, f"models/best_model_{MAP_NAME}.pth")
    elif epoch - best_model_accuracy_at_epoch >= patience:
        print(f"Early stopping at epoch {epoch+1} with best validation accuracy {best_model_accuracy:.4f} at epoch {best_model_accuracy_at_epoch+1}")
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, validation accuracy: {accuracy}')


# evaluate on test set
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    total_loss = 0
    preds = []
    labels_full = []
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        preds.extend(outputs.argmax(dim=1).cpu().numpy())
        labels_full.extend(labels.cpu().numpy())

    preds = np.array(preds)
    labels_full = np.array(labels_full)
    test_accuracy = (preds == labels_full).mean()
    print(f'Test accuracy: {test_accuracy:.4f}')