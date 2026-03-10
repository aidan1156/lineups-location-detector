from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class LineupDataset(Dataset):
    def __init__(self, map_name, transform=None):
        """
        Dataset for lineup images.
        
        Args:
            map_name: Name of the map to filter by (e.g., 'Ascent', 'Split')
            transform: Optional transform to apply to images
        """
        self.map_name = map_name
        self.transform = transform
        self.data = []
        
        lineups_df = pd.read_csv('dataset/lineups.csv')
        
        lineups_df = lineups_df[lineups_df['map'] == map_name]
        
        callouts_df = pd.read_csv(f'dataset/callouts/{map_name}.csv', header=None)
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
