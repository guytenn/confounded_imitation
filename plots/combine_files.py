import json
import os
import numpy as np

num_duplicates = 1
# min_steps = 5000000
save_name = 'Feeding_Resamp'
data_path = 'data'
files_prefixes = ['strength']
fnames = [fname for fname in os.listdir(data_path) if np.any([fname.startswith(s) for s in files_prefixes])]

name_list = []
steps_list = []
rewards_list = []
config_list = []
for fname in fnames:
    json_path = os.path.join(data_path, fname)
    with open(json_path) as f:
        data = json.load(f)  # data keys: {'config', 'rewards', 'steps'}
    for _ in range(num_duplicates):
        config_list.append(data['config'])
        rewards_list.append(data['rewards'])
        steps_list.append(data['steps'])
        name_list.append(data['config']['run_name'])
        print(fname, data['config']['run_name'])

save_dict = dict(run_names=name_list,
                 steps=steps_list,
                 rewards=rewards_list,
                 project=save_name,
                 config=config_list)

with open(os.path.join(data_path, f'{save_name}.json'), 'w') as fp:
    json.dump(save_dict, fp)

print('done')