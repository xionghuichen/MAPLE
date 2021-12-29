from .base_maple import maple_params, deepcopy

params = deepcopy(maple_params)
params.update({
    'domain': 'walker2d',
    'task': 'random-v0',
    'exp_name': 'walker2d_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/walker2d-random-v0',
    'pool_load_max_size': 10**6,
    'rollout_length': 1,
    'penalty_coeff': 1.0
})