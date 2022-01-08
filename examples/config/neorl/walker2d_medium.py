from .base_maple import maple_params, deepcopy

params = deepcopy(maple_params)
params.update({
    'domain': 'Walker2d',
    'task': 'v3',
    'exp_name': 'walker2d_neo_medium'
})
params['kwargs'].update({
    'pool_load_path': 'neorl/neorl_data/Walker2d-v3-medium-1000-train-noise.npz',
    'pool_load_max_size': 101000,
    'rollout_length': 10,
    'penalty_coeff': 0.25,
    'use_neorl': True
})