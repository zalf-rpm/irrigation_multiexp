import os
import time
from pathlib import Path

import spotpy
import spotpy_setup_MONICA
import matplotlib.pyplot as plt
import pandas as pd


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


font = {'family': 'calibri',
        'weight': 'normal',
        'size': 18}


def make_lambda(excel):
    return lambda v, p: eval(excel)


crop_sim_site_MAP = "crop_sim_site_MAP.csv"
observations = "Measurements.csv"


# read general settings
basepath = Path(os.path.dirname(os.path.abspath(__file__)))

# Read list of selected experiments we want to run the calibration for.
# In the calibration results each experimaent is a simulation, like for the first one simulation_0, for the second one simulation_1 and so on.
crop_site_map_df = pd.read_csv(os.path.join(basepath, crop_sim_site_MAP), delimiter=",")

# read observations for which the likelihood of parameter space is calculated
# Read the observations from the measurements file.
measurements_df = pd.read_csv("Measurements.csv", delimiter=";", skiprows=[1, 2])

# Read the parameters which are to be calibrated.
calib_params_df = pd.read_csv("calibratethese_ww.csv", delimiter=";")


# Here, MONICA is initialized and a producer is started:
# Arguments are: Parameters, Sites, Observations
# Returns a ready-made setup
# spot_setup = spotpy_setup_MONICA.spot_setup(params, exp_maps, obslist)
spot_setup = spotpy_setup_MONICA.SpotSetup(calib_params_df,
                                           crop_site_map_df,
                                           measurements_df)

# the same as for example: spot_setup = spot_setup(spotpy.objectivefunctions.rmse)
# Select maximum number of repititions

rep = 3 # 5000  # initial number was 10
# Set up the sampler with the model above
sampler = spotpy.algorithms.mc(spot_setup, dbname='calib_out/SCEUA_monica_results', dbformat='csv')

# Run the sampler to produce the paranmeter distribution
# and identify optimal parameters based
# on objective function
sampler.sample(rep)


def print_status_final(self, stream):
    print("\n*** Final SPOTPY summary ***", file=stream)
    print(
        "Total Duration: "
        + str(round((time.time() - self.starttime), 2))
        + " seconds"
        , file=stream)
    print("Total Repetitions:", self.rep, file=stream)

    if self.optimization_direction == "minimize":
        print("Minimal objective value: %g" % (self.objectivefunction_min), file=stream)
        print("Corresponding parameter setting:", file=stream)
        for i in range(self.parameters):
            text = "%s: %g" % (self.parnames[i], self.params_min[i])
            print(text, file=stream)

    if self.optimization_direction == "maximize":
        print("Maximal objective value: %g" % (self.objectivefunction_max), file=stream)
        print("Corresponding parameter setting:", file=stream)
        for i in range(self.parameters):
            text = "%s: %g" % (self.parnames[i], self.params_max[i])
            print(text, file=stream)

    if self.optimization_direction == "grid":
        print("Minimal objective value: %g" % (self.objectivefunction_min), file=stream)
        print("Corresponding parameter setting:", file=stream)
        for i in range(self.parameters):
            text = "%s: %g" % (self.parnames[i], self.params_min[i])
            print(text, file=stream)

        print("Maximal objective value: %g" % (self.objectivefunction_max), file=stream)
        print("Corresponding parameter setting:", file=stream)
        for i in range(self.parameters):
            text = "%s: %g" % (self.parnames[i], self.params_max[i])
            print(text, file=stream)

    print("******************************\n", file=stream)


path_to_best_out_file = "calib_out/best.out"
with open(path_to_best_out_file, "a") as _:
    print_status_final(sampler.status, _)


# Extract the parameter samples from distribution
results = spotpy.analyser.load_csv_results("calib_out/SCEUA_monica_results")

fig = plt.figure(1, figsize=(9, 5))
plt.plot(results['like1'])
plt.show()
plt.ylabel('RMSE')
plt.xlabel('Iteration')
fig.savefig('calib_out/SCEUA_objectivefunctiontrace.png', dpi=300)

bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
best_simulation = list(best_model_run[fields])

fig = plt.figure(figsize=(16, 9))
ax = plt.subplot(1, 1, 1)
ax.scatter(spot_setup.evaluation(), best_simulation, c=".3")
add_identity(ax, color='r', ls='--')
plt.xlabel('Obs')
plt.ylabel('Sim')
fig.savefig('calib_out/SCEUA_best_modelrun.png', dpi=300)
