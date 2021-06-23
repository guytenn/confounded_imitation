# Imitation Learning with Partially Observable Confouned Data

## Creating Expert Data

### Option 1: RL Agent
This option will train an  RL agent and save the data to ~/.datasets/{env_name}/{run_name}/data_1.h5
```python
python3 run.py --run_name "test_run" --env_name "rooms-v0" --algo "ppo" --n_steps 32 --n_envs 32 --save_data
```

### Option 2: Human
This will create data for rooms-v0 environment. Use a, s, d, w keys to move. Press q when done saving data. Data will be saved to ./data/expert_data.h5
```python
python3 create_human_data.py
```


## Imitation Agent
To run regular imitation with expert data saved in ~/.datasets/{env_name}/{expert_load_name}/data_1.h5
```python
python3 run.py --algo ppo --env_name "rooms-v0" --run_name "full_imitation" --expert_load_name "test_run" --n_envs 32 --n_steps 32 --dice_n_epochs 3 --dice_coeff 1 --dice_train_every 1 --eval_steps 1000
```
To run imitation with partial data add flag --partial_data