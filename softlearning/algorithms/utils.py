from copy import deepcopy


def create_SAC_algorithm(variant, *args, **kwargs):
    from .sac import SAC

    algorithm = SAC(*args, **kwargs)

    return algorithm


def create_SQL_algorithm(variant, *args, **kwargs):
    from .sql import SQL

    algorithm = SQL(*args, **kwargs)

    return algorithm

def create_MVE_algorithm(variant, *args, **kwargs):
    from .mve_sac import MVESAC

    algorithm = MVESAC(*args, **kwargs)

    return algorithm

def create_MOPO_algorithm(variant, *args, **kwargs):
    from mopo.algorithms.mopo import MOPO

    algorithm = MOPO(*args, **kwargs)

    return algorithm


ALGORITHM_CLASSES = {
    'SAC': create_SAC_algorithm,
    'SQL': create_SQL_algorithm,
    'MOPO': create_MOPO_algorithm,
}


def get_algorithm_from_variant(variant,  *args, **kwargs):
    algorithm_params = variant['algorithm_params']
    algorithm_type = algorithm_params['type']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    exp_name = variant['algorithm_params']["exp_name"]
    vae = variant['use_vae']
    retrain_model = variant['retrain_model']
    exp_name = exp_name.replace('_', '-')
    if algorithm_kwargs['separate_mean_var']:
        exp_name += '_smv'
    algorithm_kwargs["model_name"] = exp_name + '_1_{}'.format(variant['model_suffix'])
    algorithm_kwargs["tester"] = kwargs['tester']
    if variant['length'] > 0:
        algorithm_kwargs['rollout_length'] = variant['length']
    if variant['penalty_coeff'] >= 0:
        algorithm_kwargs['penalty_coeff'] = variant['penalty_coeff']
    if variant['elite_num'] > 0:
        algorithm_kwargs['num_elites'] = variant['elite_num']
    algorithm_kwargs['fix_env'] = variant['fix_env']
    kwargs = {**kwargs, **algorithm_kwargs.toDict()}
    kwargs['vae'] = variant['use_vae']
    kwargs['clip_state'] = not variant["no_clip_state"]
    kwargs['res_dyn'] = variant["res_dyn"]
    kwargs['norm_input'] = not variant["no_norm_input"]
    kwargs['seed'] = variant['run_params']['seed']
    # kwargs['load_task_name'] = variant['load_task_name']
    # kwargs['load_date'] = variant['load_date']
    print("[ DEBUG ]: kwargs to net is {}".format(kwargs))
    # if retrain_model:
    #     print('[ DEBUG ] retraining model... ')
    #     print(kwargs)
    #     kwargs['model_load_dir'] = None
    kwargs['retrain'] = retrain_model
    kwargs['network_kwargs']['embedding_size'] = variant['emb_size']
    kwargs['n_epochs'] = variant['n_epochs']
    kwargs['source'] = variant['config'].split('.')[-2]
    algorithm = ALGORITHM_CLASSES[algorithm_type](variant, *args, **kwargs)
    return algorithm
