import numpy as np
from maple.global_config import *
class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if obs.shape[-1] == 12: # In neorl, another dim is inserted to observation space.
            next_obs = next_obs[:, 1:]
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        # not_done =  np.logical_and(np.all(next_obs > -100, axis=-1),
        #                np.all(next_obs < 100, axis=-1)) * \
        not_done = np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs < STATE_CLIP_BOUND).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done
