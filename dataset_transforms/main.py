from pathlib import Path

from stages.hidden_raw_dataset import create_raw_dataset
from stages.enriched_dataset import create_enriched_dataset
from stages.cropped_dataset import create_cropped_dataset
from stages.dataset_label_send import submit_labelling_batch
from stages.dataset_label_receive import receive_labelling_results_and_create_dataset


def main():
    batch_job_path = Path("hidden/current_job_id.txt")
    if not batch_job_path.exists():
        print("No existing batch found, recomputing entire dataset")
        create_raw_dataset()
        create_enriched_dataset()
        create_cropped_dataset()
        submit_labelling_batch()

        with open("hidden/current_job_id.txt", "w") as f:
            f.write(f"{batch_job}")
    else:
        with open(batch_job_path) as f:
            batch_job = f.read().strip()

        receive_labelling_results_and_create_dataset(batch_job)
        print("done!")

if __name__ == '__main__':
    main()