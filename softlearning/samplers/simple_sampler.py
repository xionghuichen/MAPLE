from collections import defaultdict

import numpy as np

from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def initialize(self, env, policy, pool):
        super(SimpleSampler, self).initialize(env, policy, pool)
        self.get_action = self.policy[0]
        self.make_init_hidden = self.policy[1]
        self.hidden = self.make_init_hidden()

    def _process_observations(self,
                              observation,
                              action,
                              last_action,
                              reward,
                              terminal,
                              next_observation,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'last_actions': last_action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'valid': [1],
            'infos': info,
        }

        return processed_observation

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()
            #### EDIT
            if hasattr(self.env.unwrapped, "state_vector"):
                self._reset_state_vector = self.env.unwrapped.state_vector()
            ####
        lst_action = self.hidden[1]
        action, self.hidden = self.get_action(self.env.convert_to_active_observation(
                self._current_observation)[None], self.hidden)
        action = action[0]
        # print(action.shape)
        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1
        # print(lst_action.shape, lst_action.squeeze(1).shape, action.shape)
        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            last_action=lst_action.squeeze(1).squeeze(0),
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            ######## this function is siginificant for replaybuffer
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.reset_policy()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1

        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def reset_policy(self):
        self.hidden = self.make_init_hidden(1)

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
