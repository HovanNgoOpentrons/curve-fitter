import numpy as np
import json
from typing import List, Dict

# Inputs
geoID = "cuboidalWell"  # or "conicalWell"
CSV_PATH = "udv.csv"
INPUT_JSON_PATH = "3.json"
OUTPUT_JSON_PATH = "frusta_output.json"

with open(INPUT_JSON_PATH, "r") as f:
    inner_well_json = json.load(f)

depth = inner_well_json["wells"]["A1"]["depth"]
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=9, usecols=(2, 5))


def generate_frusta(data: np.ndarray, geoID: str = "conicalWell") -> List[Dict]:
    frusta_data = []

    for i in range(1, len(data)):
        vol1, h1 = data[i - 1]
        vol2, h2 = data[i]

        delta_volume = vol2 - vol1
        delta_height = h2 - h1

        if delta_height == 0:
            continue

        if geoID == "cuboidalWell":
            side_length = np.sqrt(delta_volume / delta_height)
            section = {
                "shape": geoID[:-4], 
                "bottomXDimension": side_length,
                "bottomYDimension": side_length,
                "topXDimension": side_length,
                "topYDimension": side_length,
                "topHeight": round(h2, 5),
                "bottomHeight": round(h1, 5)
            }
        elif geoID == "conicalWell":
            radius = np.sqrt(delta_volume / (np.pi * delta_height))
            diameter = 2 * radius
            section = {
                "shape": geoID[:-4],  
                "bottomDiameter": diameter,
                "topDiameter": diameter,
                "topHeight": round(h2, 5),
                "bottomHeight": round(h1, 5)
            }

        frusta_data.append(section)

    #ensure heights add up to total depth     
    last = frusta_data[-1]
    bottom_height = last["topHeight"]

    if geoID == "cuboidalWell":
        final_section = {
            "shape": geoID[:-4],
            "topXDimension": last["bottomXDimension"],
            "topYDimension": last["bottomYDimension"],
            "bottomXDimension": 0.25,
            "bottomYDimension": 0.25,
            "topHeight": depth,
            "bottomHeight": bottom_height
        }
    elif geoID == "conicalWell":
        final_section = {
            "shape": geoID[:-4],
            "topDiameter": last["bottomDiameter"],
            "bottomDiameter": 0.5,
            "topHeight": depth,
            "bottomHeight": bottom_height
        }

    frusta_data.append(final_section)

    return frusta_data

new_frusta_data = generate_frusta(data, geoID)

inner_well_json["innerLabwareGeometry"] = {
    geoID: {
        "sections": new_frusta_data
    }
}

with open(OUTPUT_JSON_PATH, "w") as f:
    json.dump(inner_well_json, f, indent=2)
