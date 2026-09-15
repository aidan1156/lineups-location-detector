"""
Send a batch of images to Gemini for labeling. 
"""

import glob
from pathlib import Path

import pandas as pd
import base64
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
from dataset_transforms.utils import Dataset, transform_dataset_path

load_dotenv()

client = genai.Client()

model_name = "gemini-3.1-flash-lite"
jsonl_filename = "hidden/image_labeling_batch.jsonl"
labels_prompt = "Analyse this image, does it contain one of the pieces of text specified in the strict JSON schema? If it does output that label exactly as specified in the schema, if it does not contain any of the specified text output null. If there is text on screen which is not exactly one of the labels, including a label in a different language, output null."

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_map_callouts() -> dict[str, list[str]]:
    map_callouts = {}
    for map_name in glob.glob('dataset_transforms/callout_conversion/*.json'):
        map_name = Path(map_name).stem
        with open(f'dataset_transforms/callout_conversion/{map_name}.json', 'r') as f:
            callouts = list(json.load(f).keys())
        map_callouts[map_name] = callouts

    return map_callouts


def submit_labelling_batch() -> str:
    map_callouts = load_map_callouts()

    print("Creating JSONL batch request file with Structured Output schemas...")
    with open(jsonl_filename, "w") as f:
        lineups_df = pd.read_csv(transform_dataset_path / f'lineups-{Dataset.CROPPED}.csv')
        for _, row in lineups_df.iterrows():

            # skip maps that don't have a schema properly defined yet
            if row["map"] not in {"Ascent", "Bind"}:
                continue

            try:
                img_path = transform_dataset_path / f'images-{Dataset.CROPPED}' / f'{row["id"]}.webp'
                map_callout = map_callouts[row["map"]]
                b64_data = encode_image_base64(img_path)
                
                payload = {
                    "id": f"image_task_{row['id']}",
                    "request": {
                        "contents": [{
                            "role": "user",
                            "parts": [
                                {"text": labels_prompt},
                                {"inline_data": {"mime_type": "image/webp", "data": b64_data}}
                            ]
                        }],
                        # This config object enforces the structured JSON output per request
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "OBJECT",
                                "properties": {
                                    "text": {
                                        "type": "STRING",
                                        "description": "The text in the image. Must be one of the specified options.",
                                        "enum": map_callout,
                                        "nullable": True
                                    },
                                },
                                "required": ["text"]
                            }
                        }
                    }
                }
                f.write(json.dumps(payload) + "\n")
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    submit_batch = input("Submit batch to Gemini API? (Y/n): ") == 'Y'

    if not submit_batch:
        print("Skipping submitting batch, exiting...")
        exit(0)

    # 2. Upload to File API
    print("Uploading batch file to Gemini...")
    uploaded_file = client.files.upload(
        file=jsonl_filename,
        config=types.UploadFileConfig(
            display_name="structured_image_batch_input",
            mime_type="application/jsonl"
        )
    )

    while True:
        file_status = client.files.get(name=uploaded_file.name)
        if file_status.state.name == "ACTIVE":
            break
        time.sleep(15)

    print("Submitting batch job...")
    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={'display_name': "structured_12k_images"}
    )

    print("Batch job created successfully")
    print(f"job name: {batch_job.name}")

    return batch_job.name