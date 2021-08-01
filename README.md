# Imitation Learning with Partially Observable Confouned Data

## Prerequisites
```
pip install numpngw screeninfo smplx trimesh
pip install 'ray[rllib]' 
pip install git+https://github.com/Zackory/bullet3.git#egg=pybullet
```

## Example

```python
python3 run_rllib.py --train --env RecSim-v2 --algo ppo --dice_coef 1 --train-timesteps 10000000 --wandb --project_name recsim --no_context --run_name 10_confounders --n_confounders 10 --covariate_shift
```
```python
python3 run_rllib.py --env RecSim-v2 --algo ppo --save-data --eval-episodes 1000 --n_confounders 7 --data_suffix conf_7
```