import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        # done = np.array([False]).repeat(len(obs))
        not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
        done = ~not_done
        done = done[:,None]
        # not_done = np.isfinite(next_obs).all(axis=-1)
        # done = done[:,None]
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        # new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        new_rew = np.clip(rew + 0.1 * np.sum(np.square(act), axis=-1), None, 3) \
                  - 0.1 * np.sum(np.square(act), axis=-1) + 15 * next_obs[..., 0]
        return new_rew
