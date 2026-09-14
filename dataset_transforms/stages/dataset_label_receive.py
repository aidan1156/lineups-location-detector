import json
import shutil
from google import genai
from google.genai import types
from dotenv import load_dotenv
import random
from pathlib import Path
from constants import Dataset, transform_dataset_path, dataset_path


load_dotenv()

client = genai.Client()
random.seed(42)


def _get_batch_results(job_name: str) -> dict[str, str]:
    batch_job = client.batches.get(name=job_name)
    print(f"Batch current state: {batch_job.state.name}")

    if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
        raise FileNotFoundError("Batch is not complete yet, results jsonl file does not exist yet")
    
    result_file_name = batch_job.dest.file_name
    
    print(f"Downloading results from: {result_file_name}")
    
    file_bytes = client.files.download(file=result_file_name)
    file_content = file_bytes.decode('utf-8')
    
    with open("hidden/batch_results.jsonl", "w") as f:
        f.write(file_content)

    id_to_label = {}
    for line in file_content.splitlines():
        data = json.loads(line)
        raw_text = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
        pred_json = json.loads(raw_text)
        prediction = str(pred_json["text"])

        id = data['id'].replace('image_task_', '')
        id_to_label[id] = prediction

    return id_to_label

def _split_dataset(id_to_callout: dict[str, str]):
    all_indicies = list(id_to_callout.keys())
    random.shuffle(all_indicies)

    train_indices = set(all_indicies[:int(len(all_indicies) * 0.8)])
    test_indices = set(all_indicies[int(len(all_indicies) * 0.8):int(len(all_indicies) * 0.9)])
    val_indices = set(all_indicies[int(len(all_indicies) * 0.9):])


    for id, callout in id_to_callout.items():
        if id in train_indices:
            folder = 'train'
        elif id in test_indices:
            folder = 'test'
        else:
            assert id in val_indices
            folder = 'val'

        # copy file from src to dst
        src = transform_dataset_path / f'images-{Dataset.CROPPED}/{id}.webp'
        dst = dataset_path / f'{folder}/{callout}/{id}.webp'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


def receive_labelling_results_and_create_dataset(job_name: str):
    id_to_callout = _get_batch_results(job_name)

    _split_dataset(id_to_callout)

    callouts = set(id_to_callout.values())
    for dataset in ['train', 'test', 'val']:
        for callout in callouts:
            callout_train_path = dataset_path / dataset / callout 
            if not callout_train_path.exists() or len(list(callout_train_path.glob('*.webp'))) == 0:
                for folder in ['test', 'val', 'train']:
                    callout_folder = dataset_path / folder / callout
                    if callout_folder.exists() and len(list(callout_folder.glob('*.webp'))) > 0:
                        # move the first image to train
                        img_path = list(callout_folder.glob('*.webp'))[0]
                        dst = dataset_path / dataset / callout / img_path.name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_path, dst)
                        break
    
