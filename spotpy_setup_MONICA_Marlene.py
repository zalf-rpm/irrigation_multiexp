import copy
from collections import defaultdict, OrderedDict
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from threading import Thread
from dateutil.relativedelta import relativedelta

import spotpy
import pandas as pd
import zmq

import monica_io3
from soil_io3 import sand_and_clay_to_ka5_texture, sand_and_clay_to_lambda

class SpotSetup(object):
    """
    This class is a setup for spotpy to calibrate MONICA model.
    """

    PATH_TO_DATA_DIR = Path("./data")

    MONICA_PATH_TO_CLIMATE_DIR  = "C:/Users/palka/GitHub/irrigation_multiexp/data"
    # MONICA_PATH_TO_CLIMATE_DIR  = "C:/Users/escueta/PycharmProjects/irrigation_multiexp/data"

    def __init__(self, user_params: pd.DataFrame, exp_maps: pd.DataFrame, obslist: pd.DataFrame):
        """
        Constructor of a spotpy setup for MONICA calibration

        :param user_params: these are the model parameter we will sample through the calibration process
        :param exp_maps: these are the experiments we want to calibrate for
        :param obslist: these are the observations we want to compare the model output to, like the real measurements
        """

        # for debug
        self.cnt_calls = 0

        self.experiments = exp_maps

        self.out = {}

        self.user_params = user_params
        self.params = []
        for index, row in self.user_params.iterrows():
            parname = row["name"]
            if not pd.isna(row["position in array"]):  # Check if position in array is not empty
                parname += "_" + str(int(row["position in array"])) # spotpy does not allow two parameters to have the same name

            if pd.isna(row["derived params"]):
            # if "derive_function" not in par: #spotpy does not care about derived params
                self.params.append(spotpy.parameter.Uniform(
                    parname,
                    row["low"],
                    row["high"],
                    row["stepsize"],
                    # row["optguess"],
                    # row["minbound"],
                    # row["maxbound"]
                ))

        self.env_templates = self._create_env()  # create the environment.json for each experiment

        # Observations data
        # add only the values for the experiments we have to do??
        self.observations = [] # for spotpy
        list_of_experiments = self.experiments['exp_ID'].values.tolist()

        # Configuration to select which observations to use
        self.config = {
            "Biomass_dm_kg_ha": False,
            "Biomass_dm_nconc_%": False,
            "Grain_dm_kg_ha": True,
            "Stem_dm_kg_ha": False,
            "Emergence": False,
            "Stem_elongation": False,
            "Anthesis": False,
            "Maturity": False,
            "SWAT_0-30_Sowing": False,
            "SWAT_30-60_Sowing": False,
            "SWAT_60-90_Sowing": False,
            "NMIN_0-30_Sowing": False,
            "NMIN_30-60_Sowing": False,
            "NMIN_60-90_Sowing": False,
            "SWAT_0-30_Anthesis": False,
            "SWAT_30-60_Anthesis": False,
            "SWAT_60-90_Anthesis": False,
            "NMIN_0-30_Anthesis": False,
            "NMIN_30-60_Anthesis": False,
            "NMIN_60-90_Anthesis": False,
            "SWAT_0-30_Harvesting": False,
            "SWAT_30-60_Harvesting": False,
            "SWAT_60-90_Harvesting": False,
            "NMIN_0-30_Harvesting": False,
            "NMIN_30-60_Harvesting": False,
            "NMIN_60-90_Harvesting": False,
        }

        params_to_columns = {
            "Biomass_dm_kg_ha": "Biomass_dm_kg_ha",
            "Biomass_dm_nconc_%": "Biomass_dm_nconc_%",
            "Grain_dm_kg_ha": "Grain_dm_kg_ha",
            "Stem_dm_kg_ha": "Stem_dm_kg_ha",
            "Emergence": "Emergence",
            "Stem_elongation": "Stem_elongation",
            "Anthesis": "Anthesis",
            "Maturity": "Maturity",
            "SWAT_0-30_Sowing": "SWAT_0-30_Sowing",
            "SWAT_30-60_Sowing": "SWAT_30-60_Sowing",
            "SWAT_60-90_Sowing": "SWAT_60-90_Sowing",
            "NMIN_0-30_Sowing": "NMIN_0-30_Sowing",
            "NMIN_30-60_Sowing": "NMIN_30-60_Sowing",
            "NMIN_60-90_Sowing": "NMIN_60-90_Sowing",
            "SWAT_0-30_Anthesis": "SWAT_0-30_Anthesis",
            "SWAT_30-60_Anthesis": "SWAT_30-60_Anthesis",
            "SWAT_60-90_Anthesis": "SWAT_60-90_Anthesis",
            "NMIN_0-30_Anthesis": "NMIN_0-30_Anthesis",
            "NMIN_30-60_Anthesis": "NMIN_30-60_Anthesis",
            "NMIN_60-90_Anthesis": "NMIN_60-90_Anthesis",
            "SWAT_0-30_Harvesting": "SWAT_0-30_Harvesting",
            "SWAT_30-60_Harvesting": "SWAT_30-60_Harvesting",
            "SWAT_60-90_Harvesting": "SWAT_60-90_Harvesting",
            "NMIN_0-30_Harvesting": "NMIN_0-30_Harvesting",
            "NMIN_30-60_Harvesting": "NMIN_30-60_Harvesting",
            "NMIN_60-90_Harvesting": "NMIN_60-90_Harvesting",
        }

        for index, row in obslist.iterrows():
            if row['Experiment'] in list_of_experiments:
                for param, column in params_to_columns.items():
                    if self.config.get(param):
                        value = row[column] if pd.notna(row[column]) else np.nan

                        if param in ["Emergence", "Stem_elongation", "Anthesis", "Maturity"]:
                            if pd.notna(value):
                                value = datetime.strptime(value, '%d.%m.%Y').timetuple().tm_yday
                            else:
                                value = np.nan

                        self.observations.append((row['Experiment'], value))

        # for simpler comparisons we sort the observations by experiment number as we do with simulations
        # and then we only keep the values
        self.observations = [v for (e, v) in sorted(self.observations, key=lambda ob: ob[0])]

        print("Final observations:", self.observations)

        self.context = zmq.Context()
        self.socket_producer = self.context.socket(zmq.PUSH)
        # self.socket_producer.connect("tcp://cluster2:6666")
        self.socket_producer.connect("tcp://localhost:6666")

        self.socket_collector = self.context.socket(zmq.PULL)
        self.socket_collector.connect("tcp://localhost:7777")

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        # the vector comes from spotpy, self.user_params holds the information coming from csv file

        # launch parallel thread for the collector
        # this is what in a simulation the consumer script would do
        collector = Thread(target=self.collect_results)
        collector.daemon = True
        collector.start()

        sim_envs = []
        # iterate over all experiments and set the model parameters and send them to MONICA
        for env in self.env_templates:
            # create a copy of the environment template
            current_env = env.copy()

            #Turn off if needed#
            # StageTemperatureSum = {}
            # BaseDaylength = {}
            # DaylengthRequirement = {}
            # VernalisationRequirement = {} # only for winter crops#

            #Turn off if needed#
            StageKcFactor = {}
            CropSpecificMaxRootingDepth = vector["CropSpecificMaxRootingDepth"]
            RootPenetrationRate = vector["RootPenetrationRate"]
            SpecificRootLength = vector["SpecificRootLength"]
            DroughtStressThreshold = {}

            #MaxAssimilationRate = vector["MaxAssimilationRate"]
            #SpecificLeafArea = {}       
            
            
            for i, name in enumerate(vector.name): #That is for the stage-specific values.
                if name.startswith("SpecificLeafArea_"):
                    SpecificLeafArea[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("StageTemperatureSum_"):
                    StageTemperatureSum[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("DroughtStressThreshold_"):
                    DroughtStressThreshold[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("StageKcFactor_"):
                    StageKcFactor[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("BaseDaylength_"):
                    BaseDaylength[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("DaylengthRequirement_"):
                    DaylengthRequirement[int(name.split('_')[1]) - 1] = vector[i]
                if name.startswith("VernalisationRequirement_"):
                    VernalisationRequirement[int(name.split('_')[1]) - 1] = vector[i]


            # exchange the values in the environment template
            #Turn off if needed#
            # for key, value in StageTemperatureSum.items():
            #     current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["StageTemperatureSum"][0][key] = value

            # for key, value in BaseDaylength.items():
            #     current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["BaseDaylength"][0][key] = value

            # for key, value in DaylengthRequirement.items():
            #     current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["DaylengthRequirement"][0][key] = value
            
            # for key, value in VernalisationRequirement.items():
            #     current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["VernalisationRequirement"][key] = value


            #Turn off if needed#
            for key, value in StageKcFactor.items():
                current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["StageKcFactor"][0][key] = value

            current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["CropSpecificMaxRootingDepth"] = CropSpecificMaxRootingDepth
            
            current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["species"]["RootPenetrationRate"] = RootPenetrationRate

            current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["species"]["SpecificRootLength"] = SpecificRootLength

            for key, value in DroughtStressThreshold.items():
                current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["DroughtStressThreshold"][key] = value
            
            #current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["MaxAssimilationRate"] = MaxAssimilationRate
            # for key, value in SpecificLeafArea.items():
            #     current_env["cropRotation"][0]["worksteps"][0]["crop"]["cropParams"]["cultivar"]["SpecificLeafArea"][0][key] = value

            sim_envs.append(current_env)

            print(f"Sending env ({self.cnt_calls}) to MONICA")
            self.socket_producer.send_json(current_env)

        # wait until the collector finishes
        collector.join()

        # build the evaluation list for spotpy
        evallist = []
        ordered_out = OrderedDict(sorted(self.out.items()))
        for k, v in ordered_out.items():
            for value in v:
                evallist.append(float(value))
        simulations = evallist

        self.cnt_calls += 1
        print(f"Call {self.cnt_calls}")
        return simulations

    def evaluation(self):
        return self.observations
        #return self.monica_model.observations

    def objectivefunction(self, simulation, evaluation):
        """
        Spotpy uses this function to calculate the error between the simulation and the evaluation data
        :param simulation:
        :param evaluation:
        :return:
        """
        objectivefunction = spotpy.objectivefunctions.rmse(evaluation,simulation) # We are using RMSE as objectiove function.
        return objectivefunction

    def collect_results(self):
        # move the socket creation to the constructor, like the one for the producer part
        # socket_collector = self.context.socket(zmq.PULL)
        # # socket_collector.connect("tcp://cluster2:7777")
        # socket_collector.connect("tcp://localhost:7777")
        received_results = 0
        leave = False

        param_extraction_config = {
            "Biomass_dm_kg_ha": lambda rec_msg: rec_msg["data"][5]['results'][0].get('AbBiom', np.nan),
            "Biomass_dm_nconc_%": lambda rec_msg: rec_msg["data"][5]['results'][0].get('AbBiomNc', np.nan),
            "Grain_dm_kg_ha": lambda rec_msg: rec_msg["data"][5]['results'][0].get('OrgBiom/Fruit', np.nan),
            "Stem_dm_kg_ha": lambda rec_msg: rec_msg["data"][5]['results'][0].get('OrgBiom/Shoot', np.nan),
            "Emergence": lambda rec_msg: self._extract_date(rec_msg, 1),
            "Stem_elongation": lambda rec_msg: self._extract_date(rec_msg, 2),
            "Anthesis": lambda rec_msg: self._extract_date(rec_msg, 3),
            "Maturity": lambda rec_msg: self._extract_date(rec_msg, 4),
            "SWAT_0-30_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('Mois', [np.nan])[0],
            "SWAT_30-60_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('Mois', [np.nan])[1],
            "SWAT_60-90_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('Mois', [np.nan])[2],
            "NMIN_0-30_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('N', [np.nan])[0],
            "NMIN_30-60_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('N', [np.nan])[1],
            "NMIN_60-90_Sowing": lambda rec_msg: rec_msg["data"][0]['results'][0].get('N', [np.nan])[2],
            "SWAT_0-30_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('Mois', [np.nan])[0],
            "SWAT_30-60_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('Mois', [np.nan])[1],
            "SWAT_60-90_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('Mois', [np.nan])[2],
            "NMIN_0-30_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('N', [np.nan])[0],
            "NMIN_30-60_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('N', [np.nan])[1],
            "NMIN_60-90_Anthesis": lambda rec_msg: rec_msg["data"][3]['results'][0].get('N', [np.nan])[2],
            "SWAT_0-30_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('Mois', [np.nan])[0],
            "SWAT_30-60_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('Mois', [np.nan])[1],
            "SWAT_60-90_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('Mois', [np.nan])[2],
            "NMIN_0-30_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('N', [np.nan])[0],
            "NMIN_30-60_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('N', [np.nan])[1],
            "NMIN_60-90_Harvesting": lambda rec_msg: rec_msg["data"][5]['results'][0].get('N', [np.nan])[2],
        }

        while not leave:
            try:
                # Start consumer here and save to json output
                rec_msg = self.socket_collector.recv_json()
            except:
                continue

            results_rec = []
            ############################################################
            try:
                for param, extractor in param_extraction_config.items():
                    if self.config.get(param):
                        try:
                            param_value = extractor(rec_msg)
                        except IndexError:
                            param_value = np.nan
                        results_rec.append(param_value)

                exp_id = rec_msg.get("customId", {}).get('exp_no', None)
                if exp_id is not None:
                    self.out[exp_id] = results_rec

            except Exception as e:
                print(f"Error processing rec_msg: {e}")
                continue

            received_results += 1
            # print("total received: " + str(received_results))
            if received_results == len(self.env_templates):
                leave = True

    def _extract_date(self, rec_msg, index):
        try:
            date_str = rec_msg["data"][index]['results'][0].get('Date', None)
            return datetime.strptime(date_str, '%Y-%m-%d').timetuple().tm_yday if date_str else np.nan
        except IndexError:
            return np.nan

    def _create_env(self, ):
        """
        This method is like a run_producer script, it creates a environment.json files for each one experiment
        but instead of sending them to MONICA, we will use them as a base environment template and set the specifyed
        model parameters via the "calibratethese" csv file.
        :return: a monica environment.json file as Python dictionary
        """
        self.env_templates = []

        # Read metadata and management data
        metadata_df = pd.read_csv(f"{self.PATH_TO_DATA_DIR}/Meta.csv", sep=';')
        fert_min_df = pd.read_csv(f"{self.PATH_TO_DATA_DIR}/Fertilisation_min.csv", sep=';')
        irrig_df = pd.read_csv(f"{self.PATH_TO_DATA_DIR}/Irrigation.csv", sep=';')
        till_df = pd.read_csv(f"{self.PATH_TO_DATA_DIR}/Management.csv", sep=';')

        # Merge datasets
        merged_df_fert_min = pd.merge(metadata_df, fert_min_df, on='Fertilisation_min')
        merged_df_irrig = pd.merge(metadata_df, irrig_df, on='Irrigation')
        merged_df_till = pd.merge(metadata_df, till_df, on='Management')

        # transform the meta dataframe to a dictionary with experiment number as key
        exp_no_to_meta = metadata_df.set_index('Experiment').T.to_dict('dict')

        for index, exp_row in self.experiments.iterrows():
            exp_id = exp_row["exp_ID"]
            with open(Path(exp_row["sim file"]), "r") as f:
                sim_json = json.load(f)
            with open(Path(exp_row["crop file"]), "r") as f:
                crop_json = json.load(f)
            with open(Path(exp_row["site file"]), "r") as f:
                site_json = json.load(f)

            # position_in_rotation = row["position in rotation"]
            crop_id = exp_row["crop ID (as in crop.json)"]

            # Extract templates from crop configuration
            fert_min_template = crop_json.pop("fert_min_template")
            irrig_template = crop_json.pop("irrig_template")
            till_template = crop_json.pop("till_template")

            # Read soil data and fill missing values
            soil_profiles = self._read_soil_data_and_fill_missing_values(f"{self.PATH_TO_DATA_DIR}/Soil.csv")

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

            meta = exp_no_to_meta[exp_id]

            # Set the crop based on the experiment
            crop_json["cropRotation"][2] = crop_id if len(crop_id) > 0 else meta['Crop']

            env_template = monica_io3.create_env_json_from_json_config({
                "crop": crop_json,
                "site": site_json,
                "sim": sim_json,
                "climate": ""  # climate_csv
            })

            # set calibration events in the env json
            # load the events from the sim.json and read the events from ['outputs']['calibration_events']
            # and write them into the env json at postion ['events']
            # monica makes no difference between calibration and normal events, it is just for setup
            try:
                env_template["events"] = sim_json["output"]["calibration_events"]
            # capture key errors and file not found errors
            except KeyError as e:
                print(
                    f"KeyError: {e}\n{"calibration_events"} not found in {sim_json}, please make sure the key in your sim.json matches!")
            except FileNotFoundError as e:
                print(
                    f"FileNotFoundError: {e}\nFile {sim_json} could not be found, double check the file name for the sim.json file!")

            worksteps: list = copy.deepcopy(env_template["cropRotation"][0]["worksteps"])

            env_template["csvViaHeaderOptions"] = sim_json["climate.csv-options"]
            env_template["pathToClimateCSV"] = (f"{self.MONICA_PATH_TO_CLIMATE_DIR}/"
                                                f"{exp_row["climate file"] if len(exp_row["climate file"]) > 0 else meta['Weather']}.csv")

            env_template["params"]["siteParameters"]["SoilProfileParameters"] = soil_profiles[meta['Soil']]

            env_template["params"]["siteParameters"]["HeightNN"] = float(meta['Elevation'])
            env_template["params"]["siteParameters"]["Latitude"] = float(meta['Lat'])

            if meta['CO2'] != 'no_co2' and not pd.isna(meta['CO2']):
                env_template["params"]["userEnvironmentParameters"]["AtmosphericCO2"] = float(meta['CO2'])

            # complete crop rotation
            dates = set()
            dates.update(exp_no_to_fertilizers[exp_id].keys(), exp_no_to_irrigation[exp_id].keys(),
                         exp_no_to_management[exp_id].keys())

            worksteps_copy = copy.deepcopy(worksteps)
            sowing_date = datetime.strptime(meta['Sowing'], '%d.%m.%Y')
            worksteps_copy[0]["date"] = sowing_date.strftime('%Y-%m-%d')
            harvest_date = datetime.strptime(meta['Harvest'], '%d.%m.%Y')
            worksteps_copy[-1]["date"] = harvest_date.strftime('%Y-%m-%d')

           # I need to update this to include a spin-up period
            start_date = sowing_date - relativedelta(months=6)
            env_template["csvViaHeaderOptions"]["start-date"] = start_date.strftime('%Y-%m-%d')

            for date in sorted(dates):
                if date in exp_no_to_fertilizers[exp_id]:
                    worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_fertilizers[exp_id][date]))
                if date in exp_no_to_irrigation[exp_id]:
                    worksteps_copy.insert(-1, copy.deepcopy(exp_no_to_irrigation[exp_id][date]))
                if date in exp_no_to_management[exp_id]:
                    tillage_event = copy.deepcopy(exp_no_to_management[exp_id][date])
                    tillage_date = datetime.strptime(tillage_event["date"], '%Y-%m-%d')
                    # Only add tillage events happening after sowing and before harvest
                    # Changed sowing_date to start_date
                    if tillage_date >= start_date and tillage_date <= harvest_date:
                        worksteps_copy.insert(-1, tillage_event)

            env_template["cropRotation"][0]["worksteps"] = worksteps_copy

            env_template["customId"] = {
                "nodata": False,
                "exp_no": exp_id,
                "soil_name": meta['Soil']
            }
            self.env_templates.append(env_template)

        return self.env_templates


    def _read_soil_data_and_fill_missing_values(self, soil_file: str):
        """

        :param soil_file:
        :return:
        """
        # Read soil data and fill missing values
        soil_df = pd.read_csv(soil_file, sep=';')
        soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']] = (soil_df[['Clay', 'Sand', 'Silt', 'pH', 'Corg']].ffill())

        soil_profiles = defaultdict(list)
        prev_depth_m = 0
        prev_soil_name = None
        # cumulative_depth = 0
        # n_per_cm = 50  # Add 50 kg N for the first 50 cm of soil depth

        for _, row in soil_df.iterrows():
            soil_name = row['Soil']
            if soil_name != prev_soil_name:
                prev_soil_name = soil_name
                prev_depth_m = 0

            current_depth_m = float(row['Depth']) / 100.0
            thickness = round(current_depth_m - prev_depth_m, 1)
            prev_depth_m = current_depth_m
            # cumulative_depth += thickness

            # Calculate nitrate for the layer
            # if cumulative_depth <= 0.5:
            #     nitrate = min(n_per_cm, thickness * 100)
            #     n_per_cm -= nitrate
            # elif n_per_cm > 0:
            #     nitrate = n_per_cm
            #     n_per_cm = 0
            # else:
            #     nitrate = 0.0

            layer = {
                "Thickness": [thickness, "m"],
                "SoilRawDensity": [float(row['Raw_density']) * 1000.0, "kg/m3"] if pd.notnull(row['Raw_density']) else
                print("Raw_density is missing for soil: ", soil_name),
                "SoilOrganicCarbon": [float(row['Corg']), "%"] if pd.notnull(row['Corg']) else print(
                    "Corg is missing for "
                    "soil: ", soil_name),
                "Clay": [float(row['Clay']), "m3/m3"],
                "Sand": [float(row['Sand']), "m3/m3"],
                "Silt": [float(row['Silt']), "m3/m3"],
            }
            soil_profiles[soil_name].append(layer)
        return soil_profiles



        