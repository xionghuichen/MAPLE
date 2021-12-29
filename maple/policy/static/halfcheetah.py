import numpy as np
from maple.global_config import *
class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        # done = np.array([False]).repeat(len(obs))
        if obs.shape[-1] == 18: # neorl
            # not_done = np.array([True]).repeat(len(obs))
            next_obs = next_obs[:, 1:]
            not_done = np.logical_and(np.all(next_obs >= -1 * STATE_CLIP_BOUND, axis=-1), np.all(next_obs <= STATE_CLIP_BOUND, axis=-1))
        else:
            not_done = np.array([True]).repeat(len(obs))
            not_done = np.logical_and(np.all(next_obs >  -1 * STATE_CLIP_BOUND, axis=-1), np.all(next_obs < STATE_CLIP_BOUND, axis=-1))
        done = ~not_done
        done = done[:,None]
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        return new_rew
