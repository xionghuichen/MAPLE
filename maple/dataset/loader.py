import os
import glob
import pickle
import gzip
import numpy as np

def restore_pool(replay_pool, experiment_root, max_size, save_path=None, adapt=False, maxlen=5,
                 policy_hook=None, fake_env=None):
    print('[ DEBUG ]: start restoring the pool')
    if 'd4rl' in experiment_root:
        restore_pool_d4rl(replay_pool, experiment_root[5:], adapt, maxlen, policy_hook, fake_env=fake_env)
    elif 'neorl' in experiment_root:
        restore_pool_d4rl(replay_pool, experiment_root[6:], adapt, maxlen, policy_hook, fake_env=fake_env)
    else:
        assert os.path.exists(experiment_root)
        if os.path.isdir(experiment_root):
            restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path)
        else:
            try:
                restore_pool_contiguous(replay_pool, experiment_root)
            except:
                restore_pool_bear(replay_pool, experiment_root)
    print('[ mbpo/off_policy ] Replay pool has size: {}'.format(replay_pool.size))


def allocate_hidden_state(replay_pool_full_traj, get_action, make_hidden):
    state = replay_pool_full_traj
    pass

def raw_data_loader(name, fake_env):
    import gym
    import d4rl
    if 'sac_data' in name:
        npz_data = np.load(f'../data/{name[9:]}')
        data = {}
        for k in npz_data.keys():
            data[k] = npz_data[k]
    elif 'neorl_data' in name:
        import neorl
        npz_data = np.load(f'../neorl_data/{name[11:]}')
        data_dict = dict(npz_data)
        data_dict["index"] = np.insert(data_dict["index"], 0, 0)
        env = neorl.make('-'.join(name[11:].split('-')[:2]))
        print("name", name[11:].split('-')[:2])
        print("env", env)
        rew_func = env.get_reward_func()
        rews = rew_func(data_dict)
        data_dict["rewards"] = rews
        data_dict['terminals'] = data_dict['done']
        data_dict['actions'] = data_dict['action']
        data_dict['observations'] = data_dict['obs']
        data_dict['next_observations'] = data_dict['next_obs']
        data = data_dict
    else:
        env = gym.make(name)
        data = d4rl.qlearning_dataset(gym.make(name))
    if 'sac_data' in name:
        data['rewards'] = fake_env.config.recompute_reward_fn(data['observations'], data['actions'], data['next_observations'],
                                            data['rewards'])
    if 'neorl_data' not in name:
        data['rewards'] = np.expand_dims(data['rewards'], axis=1)
        data['terminals'] = np.expand_dims(data['terminals'], axis=1)
    data['last_actions'] = np.concatenate((np.zeros((1, data['actions'].shape[1])), data['actions'][:-1, :]), axis=0).copy()
    # print(data['actions'] - data['last_actions'])
    data['first_step'] = np.zeros_like(data['terminals'])
    data['end_step'] = np.zeros_like(data['terminals'])
    data['valid'] = np.ones_like(data['terminals'])
    print('[ DEBUG ]: key in data: {}'.format(list(data.keys())))
    return data

def restore_pool_d4rl(replay_pool, name, adapt=False, maxlen=5, policy_hook=None, fake_env=None):
    data = raw_data_loader(name, fake_env)
    get_hidden = policy_hook if policy_hook else None
    max_traj_len = -1
    last_start = 0
    traj_num = 1
    traj_lens = []
    print('[ DEBUG ] obs shape: ', data['observations'].shape)
    ill_idxs = []
    traj_len = 0
    for i in range(data['observations'].shape[0]):
        non_terminal = True
        if i >= 1:
            non_terminal = (data['observations'][i] == data['next_observations'][i-1]).all()
            if data['terminals'][i-1]:
                non_terminal = False

        traj_len = i - last_start
        if not non_terminal:
            if data['terminals'][i - 1]:
                data['next_observations'][i - 1] = data['observations'][i - 1]
            data['last_actions'][i][:] = 0
            data['first_step'][i][:] = 1
            data['end_step'][i-1][:] = 1

            max_traj_len = max(max_traj_len, traj_len)
            last_start = i
            traj_num += 1
            traj_lens.append(traj_len)

        if traj_len > 1000:
            print('[ DEBUG + WARN ]: trajectory length is too large: current step is ', i, traj_num,)

    print(ill_idxs)
    traj_lens.append(data['observations'].shape[0] - last_start)
    assert len(traj_lens) == traj_num
    print("[ DEBUG ]: adapt is ", adapt)
    if adapt and policy_hook is not None:
        # 1, making state and lst action
        data['policy_hidden'] = None # np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
        data['value_hidden'] = None # np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
        last_start_ind = 0
        traj_num_to_infer = 400
        for i_ter in range(int(np.ceil(traj_num / traj_num_to_infer))):
            traj_lens_it = traj_lens[traj_num_to_infer * i_ter : min(traj_num_to_infer * (i_ter + 1), traj_num)]
            states = np.zeros((len(traj_lens_it), max_traj_len, data['observations'].shape[-1]))
            actions = np.zeros((len(traj_lens_it), max_traj_len, data['actions'].shape[-1]))
            lst_actions = np.zeros((len(traj_lens_it), max_traj_len, data['last_actions'].shape[-1]))
            start_ind = last_start_ind
            for ind, item in enumerate(traj_lens_it):
                states[ind, :item] = data['observations'][start_ind:(start_ind+item)]
                lst_actions[ind, :item] = data['last_actions'][start_ind:(start_ind+item)]
                actions[ind, :item] = data['actions'][start_ind:(start_ind+item)]
                start_ind += item
            print('[ DEBUG ] size of total env states: {}, actions: {}'.format(states.shape, actions.shape))
            # state, action, last_action, length
            policy_hidden_out, value_hidden_out = get_hidden(states, actions, lst_actions, np.array(traj_lens_it))
            policy_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, policy_hidden_out.shape[-1])),
                                            policy_hidden_out[:, :-1]), axis=-2)
            value_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, value_hidden_out.shape[-1])),
                                            value_hidden_out[:, :-1]), axis=-2)

            start_ind = last_start_ind
            for ind, item in enumerate(traj_lens_it):
                if data['policy_hidden'] is None:
                    data['policy_hidden'] = np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
                    data['value_hidden'] = np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
                data['policy_hidden'][start_ind:(start_ind + item)] = policy_hidden[ind, :item]
                data['value_hidden'][start_ind:(start_ind + item)] = value_hidden[ind, :item]
                start_ind += item
            last_start_ind = start_ind
    print('[ DEBUG ]: inferring hidden state done')
    data_target = {k: replay_pool.fields[k] for k in replay_pool.fields}
    traj_target_ind = 0
    mini_target_ind = 0
    mini_traj_cum_num = 0
    for k in data_target:
        data_target[k][traj_target_ind, :] = 0
    if adapt:
        for i in range(data['policy_hidden'].shape[0]):
            if mini_target_ind == 0:
                for k in data:
                    if k in data_target:
                        data_target[k][traj_target_ind, :] = 0
            for k in data:
                if k in data_target:
                    data_target[k][traj_target_ind, mini_target_ind, :] = data[k][i]
            mini_target_ind += 1
            if data['end_step'][i] or mini_target_ind == maxlen:
                traj_target_ind += 1
                traj_target_ind = traj_target_ind % replay_pool._max_size
                mini_traj_cum_num += 1
                mini_target_ind = 0
        replay_pool._pointer += mini_traj_cum_num
        replay_pool._size = int(min(mini_traj_cum_num + replay_pool._size, replay_pool._max_size))
        replay_pool._pointer %= replay_pool._max_size
        print('[ DEBUG ] data num: {}, max size: {}'.format(mini_traj_cum_num, replay_pool._max_size))

        return
    replay_pool.add_samples(data)

def reset_hidden_state(replay_pool, name, maxlen=5, policy_hook=None, fake_env=None):

    data = raw_data_loader(name, fake_env)
    get_hidden = policy_hook if policy_hook else None
    max_traj_len = -1
    last_start = 0
    traj_num = 1
    traj_lens = []
    print('[ DEBUG ] reset_hidden_state, obs shape: ', data['observations'].shape)
    for i in range(data['observations'].shape[0]):
        flag = True
        # if is_hopper_med_expert and np.max(np.abs(data['next_observations'][i] - data['observations'][i])) > 8.0 and \
        #         np.all(np.abs(data['next_observations'][i][1:]) < 5.5e-3) and i - last_start > 200:
        #     data['next_observations'][i] = data['observations'][i]
        if i >= 1:
            flag = (data['observations'][i] == data['next_observations'][i - 1]).all()
            if data['terminals'][i - 1]:
                flag = False
        if not flag:
            data['last_actions'][i, :] = 0
            data['first_step'][i, :] = 1
            data['end_step'][i - 1, :] = 1
            if data['terminals'][i - 1]:
                data['next_observations'][i-1] = data['observations'][i-1]
            traj_len = i - last_start
            max_traj_len = max(max_traj_len, traj_len)
            last_start = i
            traj_num += 1
            traj_lens.append(traj_len)
            if traj_len > 1000:
                print('[ DEBUG + WARN ] reset_hidden_state: trajectory length is too large: current step is ', i, traj_num, )
    traj_lens.append(data['observations'].shape[0] - last_start)
    # making init hidden state
    # 1, making state and lst action
    data['policy_hidden'] = None # np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
    data['value_hidden'] = None # np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
    last_start_ind = 0
    traj_num_to_infer = 400

    for i_ter in range(int(np.ceil(traj_num / traj_num_to_infer))):
        traj_lens_it = traj_lens[traj_num_to_infer * i_ter : min(traj_num_to_infer * (i_ter + 1), traj_num)]
        states = np.zeros((len(traj_lens_it), max_traj_len, data['observations'].shape[-1]))
        actions = np.zeros((len(traj_lens_it), max_traj_len, data['actions'].shape[-1]))
        lst_actions = np.zeros((len(traj_lens_it), max_traj_len, data['last_actions'].shape[-1]))
        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            states[ind, :item] = data['observations'][start_ind:(start_ind+item)]
            lst_actions[ind, :item] = data['last_actions'][start_ind:(start_ind+item)]
            actions[ind, :item] = data['actions'][start_ind:(start_ind+item)]
            start_ind += item
        print('[ DEBUG ] reset_hidden_state size of total env states: {}, actions: {}'.format(states.shape, actions.shape))
        # state, action, last_action, length
        policy_hidden_out, value_hidden_out = get_hidden(states, actions, lst_actions, np.array(traj_lens_it))
        policy_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, policy_hidden_out.shape[-1])),
                                        policy_hidden_out[:, :-1]), axis=-2)
        value_hidden = np.concatenate((np.zeros((len(traj_lens_it), 1, value_hidden_out.shape[-1])),
                                        value_hidden_out[:, :-1]), axis=-2)

        start_ind = last_start_ind
        for ind, item in enumerate(traj_lens_it):
            if data['policy_hidden'] is None:
                data['policy_hidden'] = np.zeros((data['last_actions'].shape[0], policy_hidden.shape[-1]))
                data['value_hidden'] = np.zeros((data['last_actions'].shape[0], value_hidden.shape[-1]))
            data['policy_hidden'][start_ind:(start_ind + item)] = policy_hidden[ind, :item]
            data['value_hidden'][start_ind:(start_ind + item)] = value_hidden[ind, :item]
            start_ind += item
        last_start_ind = start_ind
    print('[ DEBUG ] reset_hidden_state: inferring hidden state done')
    data_new = {'policy_hidden': data['policy_hidden'],
            'value_hidden': data['value_hidden']}
    data_adapt = {k: [] for k in data_new}
    it_traj = {k: [] for k in data_new}
    current_len = 0
    data_target = {k: replay_pool.fields[k] for k in data_new}
    traj_target_ind = 0
    mini_target_ind = 0
    for k in data_new:
        data_target[k][traj_target_ind, :] = 0
    for i in range(data_new['policy_hidden'].shape[0]):
        for k in data_new:
            data_target[k][traj_target_ind, mini_target_ind, :] = data_new[k][i]
        mini_target_ind += 1
        if data['end_step'][i] or mini_target_ind == maxlen:
            traj_target_ind += 1
            traj_target_ind = traj_target_ind % replay_pool._max_size
            mini_target_ind = 0
            for k in data_new:
                data_target[k][traj_target_ind, :] = 0



def restore_pool_softlearning(replay_pool, experiment_root, max_size, save_path=None):
    print('[ mopo/off_policy ] Loading SAC replay pool from: {}'.format(experiment_root))
    experience_paths = [
        checkpoint_dir
        for checkpoint_dir in sorted(glob.iglob(
            os.path.join(experiment_root, 'checkpoint_*')))
    ]

    checkpoint_epochs = [int(path.split('_')[-1]) for path in experience_paths]
    checkpoint_epochs = sorted(checkpoint_epochs)
    if max_size == 250e3:
        checkpoint_epochs = checkpoint_epochs[2:]

    for epoch in checkpoint_epochs:
        fullpath = os.path.join(experiment_root, 'checkpoint_{}'.format(epoch), 'replay_pool.pkl')
        print('[ mopo/off_policy ] Loading replay pool data: {}'.format(fullpath))
        replay_pool.load_experience(fullpath)
        if replay_pool.size >= max_size:
            break

    if save_path is not None:
        size = replay_pool.size
        stat_path = os.path.join(save_path, 'pool_stat_{}.pkl'.format(size))
        save_path = os.path.join(save_path, 'pool_{}.pkl'.format(size))
        d = {}
        for key in replay_pool.fields.keys():
            d[key] = replay_pool.fields[key][:size]

        num_paths = 0
        temp = 0
        path_end_idx = []
        for i in range(d['terminals'].shape[0]):
            if d['terminals'][i] or i - temp + 1 == 1000:
                num_paths += 1
                temp = i + 1
                path_end_idx.append(i)
        total_return = d['rewards'].sum()
        avg_return = total_return / num_paths
        buffer_max, buffer_min = -np.inf, np.inf
        path_return = 0.0
        for i in range(d['rewards'].shape[0]):
            path_return += d['rewards'][i]
            if i in path_end_idx:
                if path_return > buffer_max:
                    buffer_max = path_return
                if path_return < buffer_min:
                    buffer_min = path_return
                path_return = 0.0

        print('[ mopo/off_policy ] Replay pool average return is {}, buffer_max is {}, buffer_min is {}'.format(avg_return, buffer_max, buffer_min))
        d_stat = dict(avg_return=avg_return, buffer_max=buffer_max, buffer_min=buffer_min)
        pickle.dump(d_stat, open(stat_path, 'wb'))

        print('[ mopo/off_policy ] Saving replay pool to: {}'.format(save_path))
        pickle.dump(d, open(save_path, 'wb'))


def restore_pool_bear(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading BEAR replay pool from: {}'.format(load_path))
    data = pickle.load(gzip.open(load_path, 'rb'))
    num_trajectories = data['terminals'].sum() or 1000
    avg_return = data['rewards'].sum() / num_trajectories
    print('[ mopo/off_policy ] {} trajectories | avg return: {}'.format(num_trajectories, avg_return))

    for key in ['log_pis', 'data_policy_mean', 'data_policy_logvar']:
        del data[key]

    replay_pool.add_samples(data)


def restore_pool_contiguous(replay_pool, load_path):
    print('[ mopo/off_policy ] Loading contiguous replay pool from: {}'.format(load_path))
    import numpy as np
    data = np.load(load_path)

    state_dim = replay_pool.fields['observations'].shape[1]
    action_dim = replay_pool.fields['actions'].shape[1]
    expected_dim = state_dim + action_dim + state_dim + 1 + 1
    actual_dim = data.shape[1]
    assert expected_dim == actual_dim, 'Expected {} dimensions, found {}'.format(expected_dim, actual_dim)

    dims = [state_dim, action_dim, state_dim, 1, 1]
    ends = []
    current_end = 0
    for d in dims:
        current_end += d
        ends.append(current_end)
    states, actions, next_states, rewards, dones = np.split(data, ends, axis=1)[:5]
    replay_pool.add_samples({
        'observations': states,
        'actions': actions,
        'next_observations': next_states,
        'rewards': rewards,
        'terminals': dones.astype(bool)
    })
