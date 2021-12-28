from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'Hopper',
    'task': 'v3',
    'exp_name': 'hopper_neo_low'
})
params['kwargs'].update({
    'pool_load_path': 'neorl/neorl_data/Hopper-v3-low-1000-train-noise.npz',
    'pool_load_max_size': 101000,
    'rollout_length': 10,
    'penalty_coeff': 0.25,
    'use_neorl': True
})