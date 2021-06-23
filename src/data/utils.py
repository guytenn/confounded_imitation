from src.data.expert_data import ExpertData
import h5py
import numpy as np
import os


def load_expert_data(env, load_name, device):
    h5path = os.path.join(os.path.expanduser('~/.datasets'), env.get_attr('spec')[0].id, load_name, 'data_1.h5')
    data = load_h5_dataset(h5path)
    if 'dones' not in data.keys():
        data['dones'] = np.zeros(len(data['actions']))
    return ExpertData(data['states'].astype('float32'), data['actions'].astype('float32'), data['dones'], device=device)


def get_keys(h5file):
    keys = []
    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)
    h5file.visititems(visitor)
    return keys


def load_h5_dataset(h5path):
    print(f'Loading dataset in {h5path}')
    try:
        dataset_file = h5py.File(h5path, 'r')
    except:
        raise ValueError(f"No such file {h5path}")
    dataset = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()

    return dataset


if __name__ == '__main__':
    import src.envs
    import gym

    env = gym.make('SparseHopper-v0', max_x=1, max_y=1, max_z=1)

    expert_data = load_expert_data(env, 'expert_force1', device='cpu')

    debug = True