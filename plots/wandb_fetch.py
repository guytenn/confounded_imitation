import wandb
import numpy as np
import matplotlib.pyplot as plt
import traces
from scipy.interpolate import interp1d
import time
import json

save_path = 'recsim_covshift_chi.h5'

api = wandb.Api()
project_name = 'recsim-confounding-chi' # 'rl_delay', stable_baselines_pi3
# Project is specified by <entity/project-name>
# runs = api.runs("nvr-israel/{}".format(project_name))
runs = api.runs(f"guytenn/{project_name}")
summary_list = []
config_list = []
name_list = []
steps_list = []
rewards_list = []

sweep_ids = None
run_ids = [f'cov_shift_{i}' for i in range(0, 11, 2)]

for run in runs:
    if run.name in run_ids:
        h_df = run.scan_history() # can either use run.history() or, instead, scan_history() and build list of dicts via [r for r in h_df]

        steps = []
        rewards = []
        n_points = 1
        t1 = time.time()

        for i, e in enumerate(h_df):
            if i == 0:
                # run.summary are the output key/values like accuracy.
                # We call ._json_dict to omit large files
                summary_list.append(run.summary._json_dict)

                # run.config is the input metrics.
                # We remove special values that start with _.
                config = {k:v for k,v in run.config.items() if not k.startswith('_')}
                # config['sweep_id'] = run.sweep.id
                config_list.append(config)

                # run.name is the name of the run.
                name_list.append(run.name)

                n_points = run.summary._json_dict['_step'] / (e['_step'])

            t2 = time.time()
            time_diff = t2 - t1
            print(f'Estimated time to complete each run: {n_points * time_diff / 60.} minutes.')
            t1 = t2

            steps.append(e['_step'])
            rewards.append(e['reward_mean'])

        steps_list.append(steps)
        rewards_list.append(rewards)

    save_dict = dict(run_names=name_list,
                     steps=steps_list,
                     rewards=rewards_list,
                     project=project_name,
                     config=config_list)

    with open(f'{project_name}.json', 'w') as fp:
        json.dump(save_dict, fp)
