from copy import deepcopy

from . import (
    simple_replay_pool,
    extra_policy_info_replay_pool,
    union_pool,
    trajectory_replay_pool)


POOL_CLASSES = {
    'SimpleReplayPool': simple_replay_pool.SimpleReplayPool,
    'TrajectoryReplayPool': trajectory_replay_pool.TrajectoryReplayPool,
    'ExtraPolicyInfoReplayPool': (
        extra_policy_info_replay_pool.ExtraPolicyInfoReplayPool),
    'UnionPool': union_pool.UnionPool,
}

DEFAULT_REPLAY_POOL = 'SimpleReplayPool'


def get_replay_pool_from_variant(variant, env, *args, **kwargs):
    replay_pool_params = variant['replay_pool_params']
    if isinstance(replay_pool_params["kwargs"]["max_size"], int):
        replay_pool_params["kwargs"]["max_size"] = replay_pool_params["kwargs"]["max_size"]
    else:
        replay_pool_params["kwargs"]["max_size"] = replay_pool_params["kwargs"]["max_size"](variant)
    replay_pool_type = replay_pool_params['type']
    replay_pool_kwargs = deepcopy(replay_pool_params['kwargs'])
    print('[ DEBUG ]: replay pool config: ', replay_pool_kwargs)
    replay_pool = POOL_CLASSES[replay_pool_type](
        *args,
        observation_space=env.observation_space,
        action_space=env.action_space,
        **replay_pool_kwargs,
        **kwargs)

    return replay_pool
