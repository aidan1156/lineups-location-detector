"""
Read the result of a batch job from the Gemini API and save it to a JSONL file. Then, convert the JSONL results into a CSV file with the required format.
"""

import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

# 1. Read your saved job ID
with open("hidden/current_job_id.txt", "r") as f:
    job_name = f.read().strip()

# 2. Get the updated job object
batch_job = client.batches.get(name=job_name)
print(f"Current state: {batch_job.state.name}")

if batch_job.state == types.JobState.JOB_STATE_SUCCEEDED:
    # 3. Access the result file using 'dest.file_name'
    # The output is stored in a file managed by the Files API
    result_file_name = batch_job.dest.file_name
    
    print(f"Downloading results from: {result_file_name}")
    
    # 4. Download and decode
    file_bytes = client.files.download(file=result_file_name)
    file_content = file_bytes.decode('utf-8')
    
    with open("hidden/batch_results.jsonl", "w") as f:
        f.write(file_content)

    with open("dataset/lineups-in-game-callout.csv", "w") as f:
        f.write("id,prediction\n")
        for line in file_content.splitlines():
            data = json.loads(line)
            raw_text = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
            pred_json = json.loads(raw_text)
            prediction = str(pred_json["text"])

            f.write(f"{int(data['id'].replace('image_task_', '')) + 1},{prediction}\n")


else:
    print(f"Job is not finished. State: {batch_job.state.name}")