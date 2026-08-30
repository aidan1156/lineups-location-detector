# Lineups Location Detector

Training scripts for a model which detects which in-game VALORANT callout a player is standing in, based on their screen. Built for the **LineupsValorant** overlay to improve upon our automatic filtering of lineups.

## Dataset

The training images come from the lineups uploaded to **LineupsValorant** — ~11.8k screenshots spanning every map in `dataset_labelling/callout_conversion/`. Each one is stored twice: the full screenshot in `dataset/images-og/` (1000×562) and the HUD callout-text crop in `dataset/images/` (~135×64), which is what the model actually sees.

`dataset/lineups-full-image.csv` holds the source `id, map, callout` rows, and `dataset/lineups-in-game-callout.csv` holds the `id, prediction` labels produced by the Gemini pass below. The `dataset/` folder is gitignored, so only the scripts live here.

## Approach

My first attempt was a plain ResNet taking the **whole screen** as input. It didn't work, because the dataset was too sparse for it:

- Within a given callout, almost every photo framed the same landmark. Photos taken in that callout but pointed somewhere else had nothing in common with the training images and failed.
- Photos taken in one callout often contained a landmark belonging to a *different* callout, which tricked the classifier into predicting the wrong one.

The current version sidesteps this by not looking at the scene at all. It classifies a small fixed crop of the HUD region where VALORANT prints the location text (~135×64 px), so the input is essentially the callout text itself rather than the surrounding geometry.

## Pipeline

1. `dataset_labelling/dataset_labeler.py` — sends batches of cropped images to the Gemini API, which reads the callout text and labels each one against a per-map enum.
2. `dataset_labelling/read_result.py` — pulls the finished batch job back down and converts the JSONL results into a CSV.
3. `dataset_labelling/results_viewer.py` — pygame viewer for spot-checking labels.
4. `dataset_labelling/dataset_creator.py` — splits the labelled data 80/10/10 into `text-dataset/{train,test,val}/<callout>/`.
5. `train.py` / `model.py` — trains a small custom ResNet on 64×130 crops, with a weighted sampler to handle callout imbalance.
6. `convert_model.py` — exports the trained checkpoint to ONNX for use in the overlay.

`dataset_labelling/callout_conversion/` maps the raw in-game callout text to LineupsValorant's own callout names.
