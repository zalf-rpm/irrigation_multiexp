#!/usr/bin/python
# -*- coding: UTF-8

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/. */

# Authors:
# Michael Berg-Mohnicke <michael.berg@zalf.de>
#
# Maintainers:
# Currently maintained by the authors.
#
# This file has been created at the Institute of
# Landscape Systems Analysis at the ZALF.
# Copyright (C: Leibniz Centre for Agricultural Landscape Research (ZALF)

import csv
import copy
import json
import os
import sys
import zmq
from collections import defaultdict
import pandas as pd
from datetime import datetime

import monica_io3
import soil_io3
import shared
from soil_io3 import sand_and_clay_to_ka5_texture, sand_and_clay_to_lambda


def run_producer(server=None, port=None):

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)  # pylint: disable=no-member

    config = {
        "mode": "mbm-local-remote",
        "server-port": port if port else "6666",
        "server": server if server else "localhost",
        "sim.json": os.path.join(os.path.dirname(__file__), "sim_braunschweig.json"),
        "crop.json": os.path.join(os.path.dirname(__file__), "crop_braunschweig.json"),
        "site.json": os.path.join(os.path.dirname(__file__), "site_braunschweig.json"),
        "monica_path_to_climate_dir": "C:/Users/escueta/PycharmProjects/irrigation_multiexp/data_braunschweig",
        "path_to_data_dir": "./data_braunschweig/",
        "path_to_out": "out_braunschweig/",
    }
    shared.update_config(config, sys.argv, print_config=True, allow_new_keys=False)

    socket.connect("tcp://" + config["server"] + ":" + config["server-port"])

    with open(config["sim.json"]) as _:
        sim_json = json.load(_)

    with open(config["site.json"]) as _:
        site_json = json.load(_)

    with open(config["crop.json"]) as _:
        crop_json = json.load(_)

    fert_min_template = crop_json.pop("fert_min_template")
    irrig_template = crop_json.pop("irrig_template")
    till_template = crop_json.pop("till_template")

    env_template = monica_io3.create_env_json_from_json_config({
        "crop": crop_json,
        "site": site_json,
        "sim": sim_json,
        "climate": ""  # climate_csv
    })

    worksteps: list = copy.deepcopy(env_template["cropRotation"][0]["worksteps"])

    soil_df = pd.read_csv(f"{config['path_to_data_dir']}/Soil_Braunschweig.csv", sep=',|;|\t',
                          engine='python')

    # Fill missing values for clay, sand, silt, pH, corg with previous value
    soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']] = (soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']]
                                                       .ffill())

    soil_profiles = defaultdict(list)
    prev_depth_m = 0
    prev_soil_name = None
    cumulative_depth = 0

    for _, row in soil_df.iterrows():
        soil_name = row['Soil']
        if soil_name != prev_soil_name:
            prev_soil_name = soil_name
            prev_depth_m = 0

        current_depth_m = float(row['Depth']) / 100.0
        thickness = round(current_depth_m - prev_depth_m, 1)
        prev_depth_m = current_depth_m
        cumulative_depth += thickness

        layer = {
            "Thickness": [thickness, "m"],
            "PoreVolume": [float(row['Pore_volume']), "m3/m3"],
            "FieldCapacity": [float(row['Field_capacity']), "m3/m3"],
            "PermanentWiltingPoint": [float(row['Wilting_point']), "m3/m3"] if pd.notnull(row['Wilting_point']) else
            [None, "m3/m3"],
            "SoilRawDensity": [float(row['Raw_density']) * 1000.0, "kg/m3"],
            "SoilOrganicCarbon": [float(row['Corg']), "%"],
            "Clay": [float(row['Clay']), "m3/m3"],
            "Sand": [float(row['Sand']), "m3/m3"],
            "Silt": [float(row['Silt']), "m3/m3"],
            "pH": float(row['pH']),
            "KA5TextureClass": row['KA5TextureClass'] if pd.notnull(row['KA5TextureClass']) else
            sand_and_clay_to_ka5_texture(float(row['Sand']), float(row['Clay'])),
            "Lambda": sand_and_clay_to_lambda(float(row['Sand']), float(row['Clay'])),
            "SoilMoisturePercentFC": [50.0, "%"],
            # Spread 50 kg N for the first 50 cm of soil
            "SoilNitrate": [10, "kg NO3-N/m3"] if cumulative_depth <= 0.5 else [0, "kg NO3-N/m3"]
        }
        soil_profiles[soil_name].append(layer)

    metadata_df = pd.read_csv(f"{config['path_to_data_dir']}/Meta_Braunschweig.csv", sep=';|,|\t',
                              engine='python')
    fert_min_df = pd.read_csv(f"{config['path_to_data_dir']}/Fert-min_Braunschweig.csv", sep=';|,|\t',
                                engine='python')
    irrig_df = pd.read_csv(f"{config['path_to_data_dir']}/Irrigation_Braunschweig.csv", sep=';|,|\t',
                            engine='python')
    till_df = pd.read_csv(f"{config['path_to_data_dir']}/Management_Braunschweig.csv", sep=';|,|\t',
                          engine='python')

    merged_df_fert_min = pd.merge(metadata_df, fert_min_df, on='Fertilisation_min')
    merged_df_irrig = pd.merge(metadata_df, irrig_df, on='Irrigation')
    merged_df_till = pd.merge(metadata_df, till_df, on='Management')

    exp_no_to_fertilizers = defaultdict(dict)
    exp_no_to_irrigation = defaultdict(dict)
    exp_no_to_management = defaultdict(dict)

    for _, row in merged_df_fert_min.iterrows():
        fert_min_temp = copy.deepcopy(fert_min_template)
        exp_no = row['Experiment']
        fert_min_temp["date"] = row['Date']
        fert_min_temp["amount"][0] = float(row['Amount_kg_ha'])
        exp_no_to_fertilizers[exp_no][fert_min_temp["date"]] = fert_min_temp

    for _, row in merged_df_irrig.iterrows():
        irrig_temp = copy.deepcopy(irrig_template)
        exp_no = row['Experiment']
        irrig_temp["date"] = row['Date']
        irrig_temp["amount"][0] = float(row['Amount_mm'])
        exp_no_to_irrigation[exp_no][irrig_temp["date"]] = irrig_temp

    for _, row in merged_df_till.iterrows():
        till_temp = copy.deepcopy(till_template)
        exp_no = row['Experiment']
        till_temp["date"] = datetime.strptime(row['Date'], '%d.%m.%Y').strftime('%Y-%m-%d')
        till_temp["depth"] = [float(row['Depth']) / 100.0, 'm']
        exp_no_to_management[exp_no][till_temp["date"]] = till_temp

    exp_no_to_meta = metadata_df.set_index('Experiment').T.to_dict('dict')

    no_of_exps = 0

    for exp_no, meta in exp_no_to_meta.items():
        if meta['Crop'] != 'WW':
            continue

        env_template["csvViaHeaderOptions"] = sim_json["climate.csv-options"]
        env_template["pathToClimateCSV"] = f"{config['monica_path_to_climate_dir']}/{meta['Weather']}.csv"

        env_template["params"]["siteParameters"]["SoilProfileParameters"] = soil_profiles[meta['Soil']]

        env_template["params"]["siteParameters"]["HeightNN"] = float(meta['Elevation'])
        env_template["params"]["siteParameters"]["Latitude"] = float(meta['Lat'])

        env_template["params"]["userEnvironmentParameters"]["AtmosphericCO2"] = float(meta['CO2'])

        # complete crop rotation
        dates = set()
        dates.update(exp_no_to_fertilizers[exp_no].keys(), exp_no_to_irrigation[exp_no].keys(),
                     exp_no_to_management[exp_no].keys())

        worksteps_copy = copy.deepcopy(worksteps)
        worksteps_copy[0]["date"] = meta['Sowing']
        worksteps_copy[-1]["date"] = meta['Harvest']

        for date in sorted(dates):
            if date in exp_no_to_fertilizers[exp_no]:
                worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_fertilizers[exp_no][date]))
            if date in exp_no_to_irrigation[exp_no]:
                worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_irrigation[exp_no][date]))
            if date in exp_no_to_management[exp_no]:
                worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_management[exp_no][date]))

        env_template["cropRotation"][0]["worksteps"] = worksteps_copy

        env_template["customId"] = {
            "nodata": False,
            "exp_no": exp_no,
            "soil_name": meta['Soil']
        }

        socket.send_json(env_template)
        no_of_exps += 1
        print(f"{os.path.basename(__file__)} sent job {no_of_exps}")

        # Save the sent env_template as a JSON file
        with open(f"out_braunschweig/env_template_{exp_no}.json", "w") as _:
            json.dump(env_template, _, indent=4)

    # send done message
    env_template["customId"] = {
        "no_of_exps": no_of_exps,
        "nodata": True,
        "soil_name": meta['Soil']
    }
    socket.send_json(env_template)
    print(f"{os.path.basename(__file__)} done")

if __name__ == "__main__":
    run_producer()
