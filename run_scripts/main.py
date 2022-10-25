import sys
sys.path.append("../")
import os
from RLA.easy_log.tester import tester
from utils import get_parser
from maple.policy import maple
from copy import deepcopy

def get_params_from_file(filepath, params_name='params'):
    import importlib
    from dotmap import DotMap
    module = importlib.import_module(filepath)
    params = getattr(module, params_name)
    params = DotMap(params)
    return params


def get_variant_spec(command_line_args):
    from base import get_variant_spec, get_task_spec
    params = get_params_from_file(command_line_args.config)
    variant_spec = get_variant_spec(command_line_args, params)
    print(variant_spec)
    if 'neorl' in command_line_args.config:
        variant_spec['environment_params']['training']['kwargs']['use_neorl'] = True
    else:
        variant_spec['environment_params']['training']['kwargs']['use_neorl'] = False
    for k,v in vars(command_line_args).items():
        variant_spec[k] = v
    variant_spec['run_params']['seed'] = command_line_args.seed
    variant_spec = get_task_spec(variant_spec)
    return variant_spec



import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant

from softlearning.misc.utils import set_seed
import copy
import maple.policy.static as static


def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    import sys
    example_args = get_parser().parse_args(sys.argv[1:])

    variant_spec = get_variant_spec(example_args)
    # command_line_args = example_args
    print('vriant spec: {}'.format(variant_spec))

    # if command_line_args.video_save_frequency is not None:
    #     assert 'algorithm_params' in variant_spec
    #     variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
    #         command_line_args.video_save_frequency)

    variant = variant_spec
    # init
    set_seed(variant['run_params']['seed'])
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tester.set_hyper_param(**variant)
    tf.keras.backend.set_session(session)

    # build

    variant = copy.deepcopy(variant)
    # redundant code for compatibility to the older version.
    if variant['elite_num'] <= 0:
        variant['algorithm_params']['kwargs']['num_networks'] = int(variant['model_suffix'])
        variant['algorithm_params']['kwargs']['num_elites'] = int(int(variant['model_suffix']) / 7 * 5)

    if variant['loaded_task_name'] != '':
        from RLA import ExperimentLoader
        el = ExperimentLoader()
        el.config(task_name=variant['loaded_task_name'],
                  record_date=variant['loaded_date'], root='../')
        args = el.import_hyper_parameters(hp_to_overwrite=['retrain_model'])
        tester.hyper_param = vars(args)
        tester.hyper_param['retrain_model'] = False
        tester.hyper_param['algorithm_params'].kwargs.model_load_dir = variant['algorithm_params'].kwargs.model_load_dir
        variant = copy.deepcopy(tester.hyper_param)
    else:
        el = None
    tester.add_record_param(['info', "model_suffix", "penalty_coeff", "length",
                             'maple_200', 'run_params.seed', 'penalty_clip'])
    tester.configure(task_name="v2_" + variant["config"],
                     rla_config=os.path.join(get_package_path(), 'rla_config_mopo.yaml'),
                     log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()
    environment_params = variant['environment_params']
    training_environment = (get_environment_from_params(environment_params['training']))
    evaluation_environment = (get_environment_from_params(environment_params['evaluation'](variant))
        if 'evaluation' in environment_params else training_environment)

    replay_pool = (get_replay_pool_from_variant(variant, training_environment))
    sampler = get_sampler_from_variant(variant)


    #### get termination function
    domain = environment_params['training']['domain']
    static_fns = static[domain.lower()]
    ####
    if variant['elite_num'] <= 0:
        variant['algorithm_params']['kwargs']['num_networks'] = int(variant['model_suffix'])
        variant['algorithm_params']['kwargs']['num_elites'] = int(int(variant['model_suffix']) / 7 * 5)
    # construct MAPLE parameters
    algorithm_params = variant['algorithm_params']
    algorithm_kwargs = deepcopy(algorithm_params['kwargs'])
    exp_name = variant['algorithm_params']["exp_name"]
    retrain_model = variant['retrain_model']
    exp_name = exp_name.replace('_', '-')
    if algorithm_kwargs['separate_mean_var']:
        exp_name += '_smv'
    algorithm_kwargs["model_name"] = exp_name + '_1_{}'.format(variant['model_suffix'])
    kwargs = algorithm_kwargs.toDict()

    kwargs['penalty_coeff'] = variant['penalty_coeff']
    kwargs['penalty_clip'] = variant['penalty_clip']
    kwargs['rollout_length'] = variant['length']
    kwargs['seed'] = variant['run_params']['seed']
    kwargs['retrain'] = retrain_model
    kwargs['network_kwargs']['embedding_size'] = variant['emb_size']
    kwargs['n_epochs'] = variant['n_epochs']
    kwargs['source'] = variant['config'].split('.')[-2]
    kwargs['training_environment'] = training_environment
    kwargs['evaluation_environment'] = evaluation_environment
    kwargs['pool'] = replay_pool
    kwargs['static_fns'] = static_fns
    kwargs['sampler'] = sampler  # to be removed
    trainer = maple.MAPLE(**kwargs)
    if el is None:
        list(trainer.train())
    else:
        trainer.vis(el)
        trainer.performance_ns(el)

if __name__=='__main__':
    main()