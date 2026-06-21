"""
Bit messy but this script generates the response models for each map based on the callout conversion JSON files. 
It creates a Pydantic model for each map with a Literal type for the callouts in that map.
The generated models are then written to a Python file for use in the dataset labeling process.
Output is written to dataset_labelling/response_models.py
"""

import glob
from pathlib import Path
import json

map_models = []
map_models_dictionary = []
for file in glob.glob('dataset_labelling/callout_conversion/*.json'):
    map_name = Path(file).stem
    with open(file, 'r') as f:
        callouts = list(json.load(f).keys())
    map_models.append(f"""
{map_name}Callouts = Literal[{', '.join(f'"{callout}"' for callout in callouts)}]
class {map_name}CalloutResult(BaseModel):
    assigned_label: {map_name}Callouts | None
""")
    map_models_dictionary.append(f'    "{map_name}": {map_name}CalloutResult,')
    


contents = f"""
from pydantic import BaseModel
from typing import Literal

{"\n".join(map_models)}

map_models: dict[str, type[BaseModel]] = {{
{"\n".join(map_models_dictionary)}
}}
"""

with open('dataset_labelling/response_models.py', 'w') as f:
    f.write(contents)