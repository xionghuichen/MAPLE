import multiprocessing
import argparse
from distutils.util import strtobool
import json

# from ray.tune import sample_from

import softlearning.algorithms.utils as alg_utils
import softlearning.environments.utils as env_utils
from softlearning.misc.utils import datetimestamp


DEFAULT_UNIVERSE = 'gym'
DEFAULT_DOMAIN = 'HalfCheetah'
DEFAULT_TASK = 'v2'

class AlgType(object):
    MAPLE_NEORL = 'maple_neorl'
    MAPLE_D4RL = 'maple_d4rl'
    MAPLE_D4RL_200 = 'maple_d4rl_200'


TASKS_BY_DOMAIN_BY_UNIVERSE = {
    universe: {
        domain: tuple(tasks)
        for domain, tasks in domains.items()
    }
    for universe, domains in env_utils.ENVIRONMENTS.items()
}

AVAILABLE_TASKS = set(sum(
    [
        tasks
        for universe, domains in TASKS_BY_DOMAIN_BY_UNIVERSE.items()
        for domain, tasks in domains.items()
    ],
    ()))

DOMAINS_BY_UNIVERSE = {
    universe: tuple(domains)
    for universe, domains in env_utils.ENVIRONMENTS.items()
}

AVAILABLE_DOMAINS = set(sum(DOMAINS_BY_UNIVERSE.values(), ()))

UNIVERSES = tuple(env_utils.ENVIRONMENTS)

AVAILABLE_ALGORITHMS = set(alg_utils.ALGORITHM_CLASSES.keys())





def get_parser(allow_policy_list=False):
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--universe',
    #     type=str,
    #     choices=UNIVERSES,
    #     default=DEFAULT_UNIVERSE)
    # parser.add_argument(
    #     '--domain',
    #     type=str,
    #     choices=AVAILABLE_DOMAINS,
    #     default=DEFAULT_DOMAIN)
    parser.add_argument(
        '--config',
        type=str,
        default='examples.config.d4rl.halfcheetah_medium_expert'
        )
    parser.add_argument(
        '--info', type=str, default='default_info')
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--penalty_clip', type=float, default=20)
    parser.add_argument('--elite_num', type=int, default=-1)
    parser.add_argument( '--seed', type=int, default=88)
    parser.add_argument( '--n_epochs', type=int, default=1000)
    parser.add_argument(
        '--penalty_coeff', type=float, default=-1.0)
    parser.add_argument(
        '--emb_size', type=int, default=16)
    parser.add_argument(
        '--model_suffix', type=int, default=-1)
    parser.add_argument('--loaded_date', type=str, default='')
    parser.add_argument('--loaded_task_name', type=str, default='')
    parser.add_argument('--not_inherit_hp', action='store_false')
    parser.add_argument('--maple_200', action='store_true')
    parser.add_argument('--custom_config', action='store_true')
    parser.add_argument('--retrain_model', action='store_true')

    if allow_policy_list:
        parser.add_argument(
            '--policy',
            type=str,
            nargs='+',
            choices=('gaussian', ),
            default='gaussian')
    else:
        parser.add_argument(
            '--policy',
            type=str,
            choices=('gaussian', ),
            default='gaussian')



    return parser


