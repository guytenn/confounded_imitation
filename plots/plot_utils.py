import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import OrderedDict


class PlotUtils:
    @classmethod
    def plot_rewards(cls, json_path: str, title: str, max_steps: int=None, smoothing: float=10, legend_dict: OrderedDict=None):
        fig = plt.figure()
        axes = fig.add_subplot(111)

        with open(json_path) as f:
            data = json.load(f)

        cluster_ids = np.unique(data['run_names'], return_inverse=True)[1]
        clusters = [np.where(cluster_ids == i)[0] for i in range(cluster_ids.max() + 1)]

        if max_steps is None:
            max_steps = len(data['steps'][0])
        steps = data['steps'][0][1:max_steps]
        legend_values = []
        for cluster in clusters:
            if legend_dict is None:
                legend_values.append(data['run_names'][cluster[0]])
            else:
                legend_values.append(legend_dict[data['run_names'][cluster[0]]])
            cluster_rewards = np.stack([gaussian_filter1d(data['rewards'][idx][1:max_steps], sigma=smoothing)
                                        for idx in cluster if
                                        len(data['rewards'][idx][1:max_steps]) == len(steps)])
            reward_mean = np.mean(cluster_rewards, axis=0)
            reward_std = np.std(cluster_rewards, axis=0)
            axes.plot(steps, reward_mean, label=legend_values[-1])
            axes.fill_between(steps, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)

        axes.set_title(title)
        axes.set_xlabel('steps')
        axes.set_ylabel('reward')
        if legend_dict is not None:
            handles, labels = axes.get_legend_handles_labels()
            ordered_labels = list(legend_dict.values())
            ordered_handles = [handles[np.where(np.array(labels) == ordered_labels[i])[0][0]] for i in range(len(ordered_labels))]
            axes.legend(ordered_handles, ordered_labels, ncol=2, loc='lower center')
        else:
            axes.legend()
        plt.show()


if __name__ == '__main__':
    PlotUtils.plot_rewards('data/recsim-confounding-chi.json',
                           title='RecSim Confounding Effect',
                           legend_dict=OrderedDict({f'cov_shift_{i}': f'Shift Strength: {float(i / 10):g}' for i in range(0, 11, 2)}))
