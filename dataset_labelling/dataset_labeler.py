"""
Take each image in the dataset and use gemini to label it with an in game callout
"""

import pandas as pd
import base64
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from response_models import map_models

load_dotenv()

client = genai.Client()


model_name = "gemini-3.1-flash-lite"
jsonl_filename = "image_labeling_batch.jsonl"
labels_prompt = """Analyse this image, does it contain one of the pieces of text specified in the strict JSON schema? If it does output that label exactly as specified in the schema, if it does not contain any of the specified text output null."""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

print("Creating JSONL batch request file with Structured Output schemas...")
with open(jsonl_filename, "w") as f:
    lineups_df = pd.read_csv('dataset/lineups.csv')
    for idx, row in lineups_df.iterrows():
        try:
            img_path = f'dataset/images/{row["id"]}.webp'
            result_schema = map_models[row["map"]].model_json_schema()
            b64_data = encode_image_base64(img_path)
            
            payload = {
                "key": f"image_task_{idx}",
                "request": {
                    "model": model_name,
                    "contents": [{
                        "parts": [
                            {"text": labels_prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64_data}}
                        ]
                    }],
                    # This config object enforces the structured JSON output per request
                    "config": {
                        "response_mime_type": "application/json",
                        "response_schema": result_schema
                    }
                }
            }
            f.write(json.dumps(payload) + "\n")
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

# stop here for now
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

# 3. Spin up the Cloud Batch Job
print("Submitting batch job...")
batch_job = client.batches.create(
    model=model_name,
    src=uploaded_file.name,
    config={'display_name': "structured_12k_images"}
)

print("\n" + "="*60)
print("BATCH JOB CREATED SUCCESSFULLY!")
print("You can safely close your laptop or terminate this script.")
print(f"JOB_NAME_STRING: {batch_job.name}")
print("="*60)

with open("hidden/current_job_id.txt", "w") as f:
    f.write(batch_job.name)