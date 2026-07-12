"""
Convert the raw 
"""
import pandas as pd
import shutil
import random
from pathlib import Path

random.seed(42)

data = pd.read_csv('dataset/lineups-in-game-callout.csv')

all_indexes = list(data['id'].values)
random.shuffle(all_indexes)

train_indexes = set(all_indexes[:int(len(all_indexes) * 0.8)])
test_indexes = set(all_indexes[int(len(all_indexes) * 0.8):int(len(all_indexes) * 0.9)])
val_indexes = set(all_indexes[int(len(all_indexes) * 0.9):])

callouts = set(data['prediction'].values)


for index, row in data.iterrows():
    callout = row['prediction'] if not pd.isna(row['prediction']) else 'None'

    if row['id'] in train_indexes:
        folder = 'train'
    elif row['id'] in test_indexes:
        folder = 'test'
    else:
        folder = 'val'

    # copy file from src to dst
    src = Path(f'dataset/images/{row["id"]}.webp')
    dst = Path(f'text-dataset/{folder}/{callout}/{row["id"]}.webp')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)


#make sure each callout has at least 1 image in train, test, and val.
for dataset in ['train', 'test', 'val']:
    for callout in callouts:
        callout_train_path = Path(f'text-dataset/{dataset}/{callout}')
        if not callout_train_path.exists() or len(list(callout_train_path.glob('*.webp'))) == 0:
            # find an image with this callout in test or val
            for folder in ['test', 'val', 'train']:
                callout_folder = Path(f'text-dataset/{folder}/{callout}')
                if callout_folder.exists() and len(list(callout_folder.glob('*.webp'))) > 0:
                    # move the first image to train
                    img_path = list(callout_folder.glob('*.webp'))[0]
                    dst = Path(f'text-dataset/{dataset}/{callout}/{img_path.name}')
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(img_path, dst)
                    break