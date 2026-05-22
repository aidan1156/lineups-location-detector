import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance

class LineupDataset(Dataset):
    def __init__(self, map_name, transform=None):
        """
        Dataset for lineup images.
        
        Args:
            map_name: Name of the map to filter by (e.g., 'Ascent', 'Split')
            transform: Optional transform to apply to images
        """
        self.transform = transform
        self.data = []
        
        lineups_df = pd.read_csv('dataset/lineups.csv')
        
        if map_name:
            lineups_df = lineups_df[lineups_df['map'] == map_name]
            callouts_df = pd.read_csv(f'dataset/callouts/{map_name}.csv', header=None)
        else:
            import glob
            callout_files = glob.glob('dataset/callouts/*.csv')
            callouts_df = pd.concat([pd.read_csv(f, header=None) for f in callout_files], ignore_index=True)
            
        valid_callouts = set(callouts_df[0].values)
        
        for _, row in lineups_df.iterrows():
            if row['callout'] not in valid_callouts:
                continue
            
            img_path = f'dataset/images/{row["id"]}.webp'
            
            self.data.append((img_path, row['callout']))
        
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(valid_callouts))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        
        return image, label_idx
    
    def get_num_classes(self):
        return len(self.label_to_idx)
    




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