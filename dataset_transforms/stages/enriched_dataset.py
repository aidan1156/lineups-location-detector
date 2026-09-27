from PIL import Image
import pandas as pd
import shutil
from pathlib import Path
import itertools
from utils import Dataset, make_directory_writable, transform_dataset_path

enrichment_data_path = Path("enrichment-data")
# one row per enrichment image: filename,map,callout
enrichment_labels_path = enrichment_data_path / "labels.csv"

def create_enriched_dataset():
    dst_image_folder = transform_dataset_path / f'images-{Dataset.ENRICHED}'
    if dst_image_folder.exists():
        if input("Enriched dataset already exists, do you want to overwrite the existing dataset (Y/n)?: ") != 'Y':
            print("exiting...")
            exit(0)
        make_directory_writable(dst_image_folder)
        shutil.rmtree(dst_image_folder)

    shutil.copytree(transform_dataset_path / f'images-{Dataset.RAW}', transform_dataset_path / f'images-{Dataset.ENRICHED}')
    make_directory_writable(dst_image_folder)
    shutil.copy(transform_dataset_path / f'lineups-{Dataset.RAW}.csv', transform_dataset_path / f'lineups-{Dataset.ENRICHED}.csv')

    enriched_csv = transform_dataset_path / f'lineups-{Dataset.ENRICHED}.csv'
    with open(enriched_csv) as f:
        start_id = int(f.read().strip().splitlines()[-1].split(',')[0])

    ids = itertools.count(start=start_id + 1)
    labels = pd.read_csv(enrichment_labels_path, dtype=str).fillna('')
    known_maps = {p.stem for p in Path('dataset_transforms/callout_conversion').glob('*.json')}

    unlabelled = {f.name for f in enrichment_data_path.iterdir() if f.is_file() and f != enrichment_labels_path} - set(labels['filename'])
    for image in sorted(unlabelled):
        print(f"Enrichment image {image} is not in {enrichment_labels_path.name}, skipping")

    with open(enriched_csv, 'a') as f:
        for _, row in labels.iterrows():
            image, map_name, callout = row['filename'], row['map'].strip(), row['callout'].strip()
            if not map_name or not callout:
                print(f"Enrichment image {image} has no map or callout in {enrichment_labels_path.name}, skipping")
                continue
            if map_name not in known_maps:
                print(f"Enrichment image {image} has unknown map {map_name!r}, skipping")
                continue

            try:
                with Image.open(enrichment_data_path / image) as img:
                    current_id = next(ids)
                    img.save(dst_image_folder / f'{current_id}.webp', 'webp')
            except Exception as e:
                print(f"Failed to process enrichment image {image}, skipping: {e}")
                continue

            f.write(f'{current_id},{map_name},{callout}\n')
