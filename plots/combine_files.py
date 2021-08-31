import json
import os

save_name = 'Feeding_Strength'
data_path = 'data'
files_prefix = 'strength'
fnames = [fname for fname in os.listdir(data_path) if fname.startswith(files_prefix)]

name_list = []
steps_list = []
rewards_list = []
config_list = []
for fname in fnames:
    json_path = os.path.join(data_path, fname)
    with open(json_path) as f:
        data = json.load(f)  # data keys: {'config', 'rewards', 'steps'}
    config_list.append(data['config'])
    rewards_list.append(data['rewards'])
    steps_list.append(data['steps'])
    name_list.append(data['config']['run_name'])

save_dict = dict(run_names=name_list,
                 steps=steps_list,
                 rewards=rewards_list,
                 project=save_name,
                 config=config_list)

with open(os.path.join(data_path, f'{save_name}.json'), 'w') as fp:
    json.dump(save_dict, fp)