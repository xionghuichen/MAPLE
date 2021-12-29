import sys
sys.path.append("../")
import os
from RLA.easy_log.tester import tester
from run_scripts.utils import get_parser
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
    from run_scripts.base import get_variant_spec, get_task_spec
    params = get_params_from_file(command_line_args.config)
    variant_spec = get_variant_spec(command_line_args, params)
    variant_spec["info"] = command_line_args.info
    variant_spec['load_date'] = command_line_args.load_date
    variant_spec['load_task_name'] = command_line_args.load_task_name
    variant_spec['not_inherit_hp'] = command_line_args.not_inherit_hp
    variant_spec['config'] = command_line_args.config

    variant_spec['run_params']['seed'] = command_line_args.seed
    variant_spec['retrain_model'] = command_line_args.retrain_model
    variant_spec['emb_size'] = command_line_args.emb_size
    variant_spec['no_norm_input'] = False  # command_line_args.no_norm_input
    variant_spec['res_dyn'] = False  #  command_line_args.res_dyn
    variant_spec['n_epochs'] = command_line_args.n_epochs
    variant_spec['custom_config'] = command_line_args.custom_config
    variant_spec['elite_num'] = command_line_args.elite_num
    variant_spec['maple_200'] = command_line_args.maple_200
    variant_spec = get_task_spec(variant_spec)
    return variant_spec



def add_command_line_args_to_variant_spec(variant_spec, command_line_args):
    variant_spec['run_params'].update({
        'checkpoint_frequency': (
            command_line_args.checkpoint_frequency
            if command_line_args.checkpoint_frequency is not None
            else variant_spec['run_params'].get('checkpoint_frequency', 0)
        ),
        'checkpoint_at_end': (
            command_line_args.checkpoint_at_end
            if command_line_args.checkpoint_at_end is not None
            else variant_spec['run_params'].get('checkpoint_at_end', True)
        ),
    })

    variant_spec['restore'] = command_line_args.restore

    return variant_spec

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.algorithms.utils import get_algorithm_from_variant
from softlearning.policies.utils import get_policy
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.samplers.utils import get_sampler_from_variant

from softlearning.misc.utils import set_seed
import copy
import maple.policy.static as static



def _normalize_trial_resources(resources, cpu, gpu, extra_cpu, extra_gpu):
    if resources is None:
        resources = {}

    if cpu is not None:
        resources['cpu'] = cpu

    if gpu is not None:
        resources['gpu'] = gpu

    if extra_cpu is not None:
        resources['extra_cpu'] = extra_cpu

    if extra_gpu is not None:
        resources['extra_gpu'] = extra_gpu
    return resources

def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    import sys
    example_args = get_parser().parse_args(sys.argv[1:])

    variant_spec = get_variant_spec(example_args)
    # command_line_args = example_args
    print('vriant spec: {}'.format(variant_spec))

    # variant_spec = add_command_line_args_to_variant_spec(variant_spec, command_line_args)


    # if command_line_args.video_save_frequency is not None:
    #     assert 'algorithm_params' in variant_spec
    #     variant_spec['algorithm_params']['kwargs']['video_save_frequency'] = (
    #         command_line_args.video_save_frequency)

    variant = variant_spec
    # init
    set_seed(variant['run_params']['seed'])
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    tf.keras.backend.set_session(session)

    # build
    variant = copy.deepcopy(variant)

    tester.set_hyper_param(**variant)
    tester.add_record_param(['info',"model_suffix", "penalty_coeff", "length", 'run_params.seed'])
    tester.configure(task_name="v2_" + variant["config"], private_config_path=os.path.join(get_package_path(), 'rla_config_mopo.yaml'),
                     run_file='main.py', log_root=get_package_path())
    tester.log_files_gen()
    tester.print_args()
    if variant['load_task_name'] != '':
        from RLA.easy_log.tester import ExperimentLoader
        el = ExperimentLoader()
        el.config(task_name=variant['load_task_name'],
                  record_date=variant['load_date'], root='../',
                  inherit_hp=~variant['not_inherit_hp'])
        el.fork_tester_log_files()
        hp = copy.deepcopy(tester.hyper_param)
        hp['load_task_name'] = variant['load_task_name']
        hp['load_date'] = variant['load_date']
        hp['retrain_model'] = False
        hp['algorithm_params'].kwargs.model_load_dir = variant['algorithm_params'].kwargs.model_load_dir
        variant = hp
    else:
        el = None
    environment_params = variant['environment_params']
    training_environment = (get_environment_from_params(environment_params['training']))
    evaluation_environment = (get_environment_from_params(environment_params['evaluation'](variant))
        if 'evaluation' in environment_params else training_environment)

    replay_pool = (get_replay_pool_from_variant(variant, training_environment))
    sampler = get_sampler_from_variant(variant)
    # Qs = get_Q_function_from_variant(variant, training_environment)
    # policy = get_policy_from_variant(variant, training_environment, Qs)
    initial_exploration_policy = (get_policy('UniformPolicy', training_environment))

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
    kwargs['rollout_length'] = variant['length']
    kwargs['res_dyn'] = variant["res_dyn"]
    kwargs['norm_input'] = not variant["no_norm_input"]
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

if __name__=='__main__':
    main()