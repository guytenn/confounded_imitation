# Imitation Learning with Partially Observable Confouned Data

## Prerequisites
```
pip install numpngw screeninfo smplx trimesh
pip install 'ray[rllib]' 
pip install git+https://github.com/Zackory/bullet3.git#egg=pybullet
```

## Example

```python
python3 run_rllib.py --train --env RecSim-v2 --algo ppo --dice_coef 1 --wandb --project_name recsim-v2 --run_name cov_shift_5--no_context --n_confounders $shift --data_suffix conf_5; done```
```