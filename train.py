import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image, ImageEnhance
from model import Resnet
from dataset import LineupDataset

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


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def pil_to_tensor(image):
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).permute(2, 0, 1)


def random_scale(image, scale_min=0.9, scale_max=1.1):
    scale = random.uniform(scale_min, scale_max)
    width, height = image.size
    scaled_w = max(1, int(width * scale))
    scaled_h = max(1, int(height * scale))
    scaled = image.resize((scaled_w, scaled_h), Image.BILINEAR)

    if scale >= 1.0:
        left = (scaled_w - width) // 2
        upper = (scaled_h - height) // 2
        return scaled.crop((left, upper, left + width, upper + height))

    canvas = Image.new('RGB', (width, height))
    paste_x = (width - scaled_w) // 2
    paste_y = (height - scaled_h) // 2
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def random_color_jitter(image):
    brightness_factor = random.uniform(0.85, 1.15)
    contrast_factor = random.uniform(0.85, 1.15)
    saturation_factor = random.uniform(0.95, 1.05)

    image = ImageEnhance.Brightness(image).enhance(brightness_factor)
    image = ImageEnhance.Contrast(image).enhance(contrast_factor)
    image = ImageEnhance.Color(image).enhance(saturation_factor)
    return image


class TrainTransform:
    def __init__(self, output_size=(384, 216)):
        self.output_size = output_size

    def __call__(self, image):
        # Resize to slightly larger to accommodate rotation without black bars
        larger_size = (int(self.output_size[0] * 1.2), int(self.output_size[1] * 1.2))
        image = image.resize(larger_size, Image.BILINEAR)
        
        # Apply rotation on larger image
        rotation_angle = random.uniform(-12, 12)
        image = image.rotate(rotation_angle, resample=Image.BILINEAR)
        
        # Center crop to target size (removes black bars)
        width, height = image.size
        left = (width - self.output_size[0]) // 2
        top = (height - self.output_size[1]) // 2
        image = image.crop((left, top, left + self.output_size[0], top + self.output_size[1]))
        
        image = random_scale(image, scale_min=1, scale_max=1.1)
        image = random_color_jitter(image)
        return pil_to_tensor(image)


class EvalTransform:
    def __init__(self, output_size=(384, 216)):
        self.output_size = output_size

    def __call__(self, image):
        image = image.resize(self.output_size, Image.BILINEAR)
        return pil_to_tensor(image)


MAP_NAME = 'Ascent'

train_transform = TrainTransform(output_size=(384, 216))
eval_transform = EvalTransform(output_size=(384, 216))

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
    
    if accuracy > best_model_accuracy:
        best_model = model.state_dict()
        best_model_accuracy = accuracy
        best_model_accuracy_at_epoch = epoch

        # save the best model
        torch.save(best_model, f"models/best_model_{MAP_NAME}.pth")
    elif epoch - best_model_accuracy_at_epoch >= patience:
        print(f"Early stopping at epoch {epoch+1} with best validation accuracy {best_model_accuracy:.4f} at epoch {best_model_accuracy_at_epoch+1}")
        break

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, validation accuracy: {accuracy}')