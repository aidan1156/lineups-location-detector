from PIL import Image
import pandas as pd
import shutil
from pathlib import Path
import itertools
from constants import Dataset, transform_dataset_path

enrichment_data_path = Path("enrichment-data")

def create_enriched_dataset():
    dst_image_folder = transform_dataset_path / f'images-{Dataset.ENRICHED}'
    if dst_image_folder.exists():
        if input("Enriched dataset already exists, do you want to overwrite the existing dataset (Y/n)?: ") != 'Y':
            print("exiting...")
            exit(0)
        shutil.rmtree(dst_image_folder)

    shutil.copytree(transform_dataset_path / f'images-{Dataset.RAW}', transform_dataset_path / f'images-{Dataset.ENRICHED}')
    shutil.copy(transform_dataset_path / f'lineups-{Dataset.RAW}.csv', transform_dataset_path / f'lineups-{Dataset.ENRICHED}.csv')

    enriched_csv = transform_dataset_path / f'lineups-{Dataset.ENRICHED}.csv'
    with open(enriched_csv) as f:
        start_id = int(f.read().strip().splitlines()[-1].split(',')[0])

    ids = itertools.count(start=start_id + 1)
    images = [f.name for f in enrichment_data_path.iterdir() if f.is_file()]
    with open(enriched_csv, 'a') as f:
        for image in images:
            try:
                with Image.open(enrichment_data_path / image) as img:
                    current_id = next(ids)
                    img.save(dst_image_folder / f'{current_id}.webp', 'webp')
            except Exception as e:
                print(f"Failed to process enrichment image {image}, skipping: {e}")
                continue

            # enriched data is map agnostic, so just hoy it in as Ascent
            f.write(f'{current_id},Ascent,A Site\n')
