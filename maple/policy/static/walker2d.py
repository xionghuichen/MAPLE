import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        if obs.shape[-1] == 18: # In neorl, another dim is inserted to observation space.
            # not_done = np.array([True]).repeat(len(obs))
            next_obs = next_obs[:, 1:]
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                    * (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done
