
import numpy as np
import pdb

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'Point2DWallEnv': 50,
    'Pendulum': 200,
}
import tensorflow as tf
import os
def get_package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALGORITHM_PARAMS_ADDITIONAL = {
    'MAPLE': {
        'type': 'MAPLE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(5000),
            "model_load_dir": os.path.join(get_package_path(), 'models'),
            "num_networks": 7,
            "network_kwargs": {
                "hidden_sizes": [256, 256],
                "activation": tf.nn.relu,
                "output_activation": None,
                "lstm_hidden_unit": 128,
                "embedding_size": 16
            }
        }
    },
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Hopper': int(1e3),
    'HalfCheetah': int(1e3),
    'Walker2d': int(1e3),
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(domain, DEFAULT_MAX_PATH_LENGTH) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}


NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm, env_params):
    print("get algorithms", algorithm)
    algorithm_params = deep_update(
        env_params,
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    algorithm_params = deep_update(
        algorithm_params,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    )
    variant_spec = {
        # 'git_sha': get_git_rev(),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': {},
            },
            'evaluation': lambda spec: (
                spec['environment_params']['training']),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': lambda spec: (
                    {
                        'SimpleReplayPool': int(1e6),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(spec['replay_pool_params']['type'], int(1e6))
                ),
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': 88,
            'checkpoint_at_end': True,
            'checkpoint_frequency': NUM_EPOCHS_PER_DOMAIN.get(
                domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
            'info': ''
        },
    }

    return variant_spec

def get_variant_spec(args, env_params):
    universe, domain, task = env_params.universe, env_params.domain, env_params.task
    variant_spec = get_variant_spec_base(
        universe, domain, task, args.policy, env_params.type, env_params)
    return variant_spec

NEORL_CONFIG = {
    "hopper":
        {
            'common': {
                'length': 10,
                'penalty_coeff': 1.0,
            },
        },
    "halfcheetah":
        {
            'common': {
                'penalty_clip': 4,
                'length': 15,
                'penalty_coeff': 1.0,
            }
        },
    'walker2d':
        {
            'common': {
                'length': 15,
                'penalty_coeff': 0.25,
            }
        }
}
D4RL_MAPLE_CONFIG = {
    'common':{
        'length': 10,
        'penalty_coeff': 0.25,
    },
    'halfcheetah':{
        'common': {},
        'medium-expert':
            {
                'n_epochs': 2000,
                'penalty_coeff': 5.0,
            }
    }
}

D4RL_MAPLE_200_CONFIG = {
    'common': {
        'length': 20,
        'penalty_coeff': 0.25,
    },
    'halfcheetah': {
        'common': {},
        'medium-expert':
            {
                'n_epochs': 2000,
                'length': 10,
                'penalty_coeff': 5.0,
            },
        'mixed': {
            'penalty_clip': 4.0,
        }
    },
    'hopper': {
        'common': {
            'penalty_coeff': 1.0,
        }
    },
}
def get_task_spec(variant_spec):
    if variant_spec["custom_config"]:
        return variant_spec
    else:
        if variant_spec['environment_params']['training']['kwargs']['use_neorl']:
            if variant_spec['maple_200']:
                assert "have not test maple_200 in neorl yet"
            variant_spec['model_suffix'] = 50
            tasks = variant_spec['config'].split('.')[-1].split('_')
            variant_spec.update(NEORL_CONFIG[tasks[0]]['common'])
        else:
            tasks = variant_spec['config'].split('.')[-1].split('_')
            if variant_spec['maple_200']:
                variant_spec['model_suffix'] = 200
                config = D4RL_MAPLE_200_CONFIG
            else:
                variant_spec['model_suffix'] = 20
                config = D4RL_MAPLE_CONFIG
            variant_spec.update(config['common'])
            if tasks[0] in config.keys():
                variant_spec.update(config[tasks[0]]['common'])
                behavior_type = '-'.join(tasks[1:])
                if behavior_type in config[tasks[0]].keys():
                    variant_spec.update(config[tasks[0]][behavior_type])
        return variant_spec
