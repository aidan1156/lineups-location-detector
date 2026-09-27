from PIL import Image
import pandas as pd
import shutil
from utils import Dataset, make_directory_writable, transform_dataset_path


def create_cropped_dataset():
    data = pd.read_csv(transform_dataset_path / f'lineups-{Dataset.ENRICHED}.csv')

    dst_folder = transform_dataset_path / f"images-{Dataset.CROPPED}"

    if dst_folder.exists():
        if input("Cropped dataset already exists, do you want to overwrite the existing dataset (Y/n)?: ") != 'Y':
            print("exiting...")
            exit(0)
        make_directory_writable(dst_folder)
        shutil.rmtree(dst_folder)

    dst_folder.mkdir()
    make_directory_writable(dst_folder)

    with open(transform_dataset_path / f"lineups-{Dataset.CROPPED}.csv", "w") as f:
        f.write("id,map\n")
        for i, row in data.iterrows():
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(data)} images")
            src_path = transform_dataset_path / f"images-{Dataset.ENRICHED}" / f"{row["id"]}.webp"

            dst_path = dst_folder / f"{row["id"]}.webp"
            try:
                with Image.open(src_path) as img:
                    width, height = img.size
                    left = int(width * 0.08)
                    right = int(width * 0.15)
                    top = int(height * 0.00)
                    bottom = int(height * 0.06)
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_img.save(dst_path)
            except Exception as e:
                print(f"Failed to process image {row["id"]}, skipping: {e}")
                continue

            f.write(f"{row["id"]},{row["map"]}\n")

