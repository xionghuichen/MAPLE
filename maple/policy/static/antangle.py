import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        x = next_obs[:, 0]
        not_done = 	np.isfinite(next_obs).all(axis=-1) \
        			* (x >= 0.2) \
        			* (x <= 1.0)
        not_done2 = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
        not_done = np.logical_and(not_done2, not_done)
        done = ~not_done
        done = done[:,None]
        return done

    @staticmethod
    def recompute_reward_fn(obs, act, next_obs, rew):
        survive_reward = 1
        ctrl_cost = .5 * np.square(act).sum(axis=-1)
        xy_velocity = next_obs[..., 111:]
        contact_cost = 0.5 * 1e-3 * np.sum((np.square(next_obs[..., 27:111])), axis=-1)
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        new_rew = xy_velocity[..., 0] * np.cos(np.pi/6) + xy_velocity[..., 1] * np.sin(np.pi/6) - ctrl_cost - contact_cost + survive_reward
        # new_rew = -(rew + 0.1 * np.sum(np.square(act))) - 0.1 * np.sum(np.square(act))
        return new_rew

