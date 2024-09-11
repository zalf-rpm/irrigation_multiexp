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

import copy
import json
import os
import sys
import zmq
from collections import defaultdict
import pandas as pd
from datetime import datetime

import monica_io3
import shared
from soil_io3 import sand_and_clay_to_ka5_texture, sand_and_clay_to_lambda


def run_producer(server=None, port=None):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)  # pylint: disable=no-member

    config = {
        "mode": "mbm-local-remote",
        "server-port": port if port else "6666",
        "server": server if server else "localhost",
        "sim.json": os.path.join(os.path.dirname(__file__), "sim.json"),
        "crop.json": os.path.join(os.path.dirname(__file__), "crop.json"),
        "site.json": os.path.join(os.path.dirname(__file__), "site.json"),
        "monica_path_to_climate_dir": "C:/Users/palka/GitHub/irrigation_multiexp/data",
        "path_to_data_dir": "./data/",
        "path_to_out": "out/",
    }
    shared.update_config(config, sys.argv, print_config=True, allow_new_keys=False)

    socket.connect("tcp://" + config["server"] + ":" + config["server-port"])

    with open(config["sim.json"]) as _:
        sim_json = json.load(_)

    with open(config["site.json"]) as _:
        site_json = json.load(_)

    with open(config["crop.json"]) as _:
        crop_json = json.load(_)

    # Extract templates from crop configuration
    fert_min_template = crop_json.pop("fert_min_template")
    irrig_template = crop_json.pop("irrig_template")
    till_template = crop_json.pop("till_template")

    # Read soil data and fill missing values
    soil_df = pd.read_csv(f"{config['path_to_data_dir']}/Soil.csv", sep=';')
    soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']] = (soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']].ffill())

    soil_profiles = defaultdict(list)
    prev_depth_m = 0
    prev_soil_name = None
    cumulative_depth = 0
    n_per_cm = 50  # Add 50 kg N for the first 50 cm of soil depth

    for _, row in soil_df.iterrows():
        soil_name = row['Soil']
        if soil_name != prev_soil_name:
            prev_soil_name = soil_name
            prev_depth_m = 0

        current_depth_m = float(row['Depth']) / 100.0
        thickness = round(current_depth_m - prev_depth_m, 1)
        prev_depth_m = current_depth_m
        cumulative_depth += thickness

        # Calculate nitrate for the layer
        if cumulative_depth <= 0.5:
            nitrate = min(n_per_cm, thickness * 100)
            n_per_cm -= nitrate
        elif n_per_cm > 0:
            nitrate = n_per_cm
            n_per_cm = 0
        else:
            nitrate = 0.0

        layer = {
            "Thickness": [thickness, "m"],
            "PoreVolume": [float(row['Pore_volume']), "m3/m3"] if pd.notnull(row['Pore_volume']) else [None, "m3/m3"],
            "FieldCapacity": [float(row['Field_capacity']), "m3/m3"] if pd.notnull(row['Field_capacity']) else
            [None, "m3/m3"],
            "PermanentWiltingPoint": [float(row['Wilting_point']), "m3/m3"] if pd.notnull(row['Wilting_point']) else
            [None, "m3/m3"],
            "SoilRawDensity": [float(row['Raw_density']) * 1000.0, "kg/m3"] if pd.notnull(row['Raw_density']) else
            print("Raw_density is missing for soil: ", soil_name),
            "SoilOrganicCarbon": [float(row['Corg']), "%"] if pd.notnull(row['Corg']) else print("Corg is missing for "
                                                                                                 "soil: ", soil_name),
            "Clay": [float(row['Clay']), "m3/m3"],
            "Sand": [float(row['Sand']), "m3/m3"],
            "Silt": [float(row['Silt']), "m3/m3"],
            "pH": float(row['pH']) if pd.notnull(row['pH']) else None,
            "KA5TextureClass": sand_and_clay_to_ka5_texture(float(row['Sand']), float(row['Clay'])),
            "Lambda": sand_and_clay_to_lambda(float(row['Sand']), float(row['Clay'])),
            "SoilMoisturePercentFC": [50.0, "%"],
            "SoilNitrate": [nitrate, "kg/ha"],
        }
        soil_profiles[soil_name].append(layer)

    # View all soil profiles
    # for soil_name, layers in soil_profiles.items():
    #     print(soil_name)
    #     for layer in layers:
    #         print(layer)

    # Read metadata and management data
    metadata_df = pd.read_csv(f"{config['path_to_data_dir']}/Meta.csv", sep=';')
    fert_min_df = pd.read_csv(f"{config['path_to_data_dir']}/Fertilisation_min.csv", sep=';')
    irrig_df = pd.read_csv(f"{config['path_to_data_dir']}/Irrigation.csv", sep=';')
    till_df = pd.read_csv(f"{config['path_to_data_dir']}/Management.csv", sep=';')

    # Merge datasets
    merged_df_fert_min = pd.merge(metadata_df, fert_min_df, on='Fertilisation_min')
    merged_df_irrig = pd.merge(metadata_df, irrig_df, on='Irrigation')
    merged_df_till = pd.merge(metadata_df, till_df, on='Management')

    exp_no_to_fertilizers = defaultdict(dict)
    exp_no_to_irrigation = defaultdict(dict)
    exp_no_to_management = defaultdict(dict)

    for _, row in merged_df_fert_min.iterrows():
        if pd.isna(row['Fertilisation_min']) or row['Fertilisation_min'] == 'no_fert':
            continue
        fert_min_temp = copy.deepcopy(fert_min_template)
        fert_min_temp["date"] = datetime.strptime(row['Date'], '%d.%m.%Y').strftime('%Y-%m-%d')
        fert_min_temp["amount"][0] = float(row['Amount_kg_ha'])
        exp_no_to_fertilizers[row['Experiment']][fert_min_temp["date"]] = fert_min_temp

    for _, row in merged_df_irrig.iterrows():
        if pd.isna(row['Irrigation']) or row['Irrigation'] in ['no_irrig', 'wet', 'dry', 1, 2]:
            continue
        irrig_temp = copy.deepcopy(irrig_template)
        irrig_temp["date"] = datetime.strptime(row['Date'], '%d.%m.%Y').strftime('%Y-%m-%d')
        irrig_temp["amount"][0] = float(row['Amount_mm'])
        exp_no_to_irrigation[row['Experiment']][irrig_temp["date"]] = irrig_temp

    for _, row in merged_df_till.iterrows():
        if pd.isna(row['Management']) or row['Management'] == 'no_manag':
            continue
        till_temp = copy.deepcopy(till_template)
        till_temp["date"] = datetime.strptime(row['Date'], '%d.%m.%Y').strftime('%Y-%m-%d')
        till_temp["depth"] = [float(row['Depth']) / 100.0, 'm']
        exp_no_to_management[row['Experiment']][till_temp["date"]] = till_temp

    exp_no_to_meta = metadata_df.set_index('Experiment').T.to_dict('dict')
    no_of_exps = 0

    for exp_no, meta in exp_no_to_meta.items():
        # Skip experiments with these crops
        if meta['Crop'] in ['PO', 'WC', 'ZU', 'TR']:
            continue
        # Crops for simulation: WW, WB, SB, WR, SM, SW
        if (meta['Crop'] != 'WW' or pd.isna(meta['Sowing'] or pd.isna(meta['Harvest'])) or meta['Name'] in
        # if (pd.isna(meta['Sowing'] or pd.isna(meta['Harvest'])) or meta['Name'] in
                ['JKI_Braunschweig_Rainshelter',
                 'UTP_Bydgoszcz',
                 'FI_Dahlhausen',
                 #'ATB_Marquart',
                 #'TI_Braunschweig_FACE',
                 #'ZALF_Muencheberg_V4',
                 #'HUB_Thyrow_D1'
                 ]):
            continue

        # Set the crop based on the experiment
        crop_json["cropRotation"][2] = meta['Crop']

        env_template = monica_io3.create_env_json_from_json_config({
            "crop": crop_json,
            "site": site_json,
            "sim": sim_json,
            "climate": ""  # climate_csv
        })

        worksteps: list = copy.deepcopy(env_template["cropRotation"][0]["worksteps"])

        env_template["csvViaHeaderOptions"] = sim_json["climate.csv-options"]
        env_template["pathToClimateCSV"] = f"{config['monica_path_to_climate_dir']}/{meta['Weather']}.csv"

        env_template["params"]["siteParameters"]["SoilProfileParameters"] = soil_profiles[meta['Soil']]

        env_template["params"]["siteParameters"]["HeightNN"] = float(meta['Elevation'])
        env_template["params"]["siteParameters"]["Latitude"] = float(meta['Lat'])

        if meta['CO2'] != 'no_co2' and not pd.isna(meta['CO2']):
            env_template["params"]["userEnvironmentParameters"]["AtmosphericCO2"] = float(meta['CO2'])

        # complete crop rotation
        dates = set()
        dates.update(exp_no_to_fertilizers[exp_no].keys(), exp_no_to_irrigation[exp_no].keys(),
                     exp_no_to_management[exp_no].keys())

        worksteps_copy = copy.deepcopy(worksteps)
        sowing_date = datetime.strptime(meta['Sowing'], '%d.%m.%Y')
        worksteps_copy[0]["date"] = sowing_date.strftime('%Y-%m-%d')
        harvest_date = datetime.strptime(meta['Harvest'], '%d.%m.%Y')
        worksteps_copy[-1]["date"] = harvest_date.strftime('%Y-%m-%d')

        for date in sorted(dates):
            if date in exp_no_to_fertilizers[exp_no]:
                worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_fertilizers[exp_no][date]))
            if date in exp_no_to_irrigation[exp_no]:
                worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_irrigation[exp_no][date]))
            if date in exp_no_to_management[exp_no]:
                tillage_event = copy.deepcopy(exp_no_to_management[exp_no][date])
                tillage_date = datetime.strptime(tillage_event["date"], '%Y-%m-%d')
                # Only add tillage events happening after sowing and before harvest
                if tillage_date >= sowing_date and tillage_date <= harvest_date:
                    worksteps_copy.insert(-1, tillage_event)

        env_template["cropRotation"][0]["worksteps"] = worksteps_copy

        env_template["customId"] = {
            "nodata": False,
            "exp_no": exp_no,
            "soil_name": meta['Soil']
        }

        socket.send_json(env_template)
        no_of_exps += 1
        print(f"{os.path.basename(__file__)} sent job {no_of_exps} for experiment number: {exp_no}")

        # Save the sent env_template as a json file
        # with open(f"out/env_template_{exp_no}.json", "w") as _:
        #     json.dump(env_template, _, indent=4)

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
