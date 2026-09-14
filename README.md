# Lineups Location Detector

Training scripts for a model which detects which in-game VALORANT callout a player is standing in, based on their screen. Built for the **LineupsValorant** overlay to improve upon our automatic filtering of lineups.

## Dataset

The training images come from the lineups uploaded to **LineupsValorant**. Screenshots come in at whatever resolution the uploader plays at — nothing in the pipeline assumes a fixed size, and everything is stored as `.webp`.

The dataset is built in stages. The first three write to `intermediate-datasets/` as an `images-<stage>/` folder plus a matching `lineups-<stage>.csv`; the labelling stage turns the last of those into the dataset the model trains on:

| Stage | Output | Contents |
| --- | --- | --- |
| `raw` | `images-raw/`, `lineups-raw.csv` | Full screenshots pulled from the uploaded lineups, with `id, map, callout` rows. |
| `enriched` | `images-enriched/`, `lineups-enriched.csv` | The raw set plus anything dropped into `enrichment-data/`, appended with fresh ids. |
| `cropped` | `images-cropped/`, `lineups-cropped.csv` | The HUD callout-text crop of each image — what the model actually sees. |
| labelling | `dataset/<split>/<callout>/` | Gemini reads the callout text off each crop against that map's list of callouts. Results are split 80/10/10 and filed under the callout it read; crops it couldn't match any callout to land under `None`. |

`intermediate-datasets/`, `dataset/` and the `hidden/` folder holding the batch job id are all gitignored, so only the scripts live here.

The crop is taken as a **proportion** of each image rather than a fixed pixel box: horizontally from 8% to 15% of the width, vertically from the top edge down to 6% of the height. On a 1920×1080 screenshot that works out to roughly 134×65 px, but a 2560×1440 or ultrawide screenshot crops to the same region of the HUD at its own size. Images are only resized to a fixed shape at training time, where the transform pipeline scales them to 64 px tall and centre-crops to 64×130.

## Approach

My first attempt was a plain ResNet taking the **whole screen** as input. It didn't work, because the dataset was too sparse for it:

- Within a given callout, almost every photo framed the same landmark. Photos taken in that callout but pointed somewhere else had nothing in common with the training images and failed.
- Photos taken in one callout often contained a landmark belonging to a *different* callout, which tricked the classifier into predicting the wrong one.

The current version sidesteps this by not looking at the scene at all. It classifies the small region of the HUD where VALORANT prints the location text, so the input is essentially the callout text itself rather than the surrounding geometry.

## Pipeline

`dataset_transforms/main.py` drives the whole thing. Because the labelling step is a Gemini *batch* job that takes a while to come back, the run is split in two by `hidden/current_job_id.txt`:

- **No job file** — rebuild the dataset from scratch (raw → enriched → cropped), submit the labelling batch, and record the job id.
- **Job file present** — pull that batch's results down and build the final split dataset.

The stages live in `dataset_transforms/stages/`:

1. `hidden_raw_dataset.py` — fetches the uploaded lineups into the `raw` stage. Gitignored (`hidden_*`), so it isn't in this repo.
2. `enriched_dataset.py` — copies the raw stage forward and appends the extra images sitting in `enrichment-data/`. These are map-agnostic, so they're just filed under Ascent.
3. `cropped_dataset.py` — crops every image to the HUD callout-text region described above, skipping anything that fails to open.
4. `dataset_label_send.py` — base64s the cropped images into a JSONL batch request and submits it to the Gemini API. Each request carries a structured-output schema whose enum is that map's callout list, so the model either returns one of the exact callout strings or `null`. Only maps with a finished schema are included — currently Ascent and Bind.
5. `dataset_label_receive.py` — downloads the finished batch, parses the predictions, and splits them 80/10/10 into `dataset/{train,test,val}/<callout>/`. Images the labeller returned `null` for end up under a `None` class. A final pass copies an image into any split/callout combination that would otherwise be empty, so no class disappears from a split.

`dataset_transforms/callout_conversion/<Map>.json` maps each raw in-game callout string (the key, and what Gemini is asked to read) to the LineupsValorant callout names it corresponds to (the value, a list — one in-game region can cover several of our callouts).

## Training

The training scripts live in `training/`, and everything they produce — checkpoints, ONNX exports, metadata — goes in `training/model/`.

- `training/model.py` — a small custom ResNet (a 7×7 stem then four 2-block stages, 64→512 channels).
- `training/train.py` — trains it on 64×130 crops with light random affine jitter, Adamax + cosine annealing, and a weighted sampler that draws a balanced epoch so rare callouts aren't drowned out. Runs up to 100 epochs with early stopping after 10 without improvement, writing checkpoints and `training_metadata.json` to `training/model/`.
- `training/test.py` — evaluates `training/model/best_model.pt` against the test split.
- `training/convert_model.py` — exports the trained checkpoint to ONNX for use in the overlay.

Run them from the repo root (`python training/train.py`) — the dataset and checkpoint paths are all relative to it.
