import os
import random
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
import torch
from scipy.stats import loguniform


def right_hand_side(t, x, ts, us, values, rhs_func):
    """
    Calculate the rhs of the integral, at time t, state x.
    ts, us are used to get the current input.
    values are constants in the integral.
    """

    # Interp is needed to get u for timestep
    u1 = np.interp(t, ts, us[:, 0])
    u2 = np.interp(t, ts, us[:, 1])

    arguments = np.hstack((x, u1, u2, values))

    # Need call to array and reshape, as solve_ipv requires state vector to be 1d
    dx = np.array(rhs_func(*arguments))

    return dx.reshape(dx.size)


def seed_everything(seed):
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(env, algo, episodes=10, use_robust_rl=False):
    import tqdm
    stat_dict = {}
    trange_ = tqdm.trange(episodes)
    for episode in trange_:
        obs, info = env.reset(episode)
        # algo.model.log_dict_for_adaptor = {"ground_truth": [],
        #                                    "predicted": []} if use_robust_rl else None
        action = env.action_space.sample() * 0.
        episode_stat = {}
        done = False
        lstm_cell_size = algo.config["model"]["lstm_cell_size"]
        state = [np.zeros([lstm_cell_size], np.float32) for _ in range(2)] if lstm_cell_size is not None else None
        while not done:
            episode_stat["states"] = [obs] if "states" not in episode_stat else episode_stat["states"] + [obs]
            episode_stat["actions"] = [action] if "actions" not in episode_stat else \
                episode_stat["actions"] + [action]
            episode_stat["infos"] = [info] if "infos" not in episode_stat else episode_stat["infos"] + [info]
            # episode_stat["log_dict_for_adaptor"] = [] if use_robust_rl else None

            input_dict = {"obs": obs}

            input_dict.update({
                "obs": obs,
                "state_in_0": state[0],
                "state_in_1": state[1],
                "prev_actions": np.stack(episode_stat["actions"][-1]),
            })

            action, state, action_info = algo.compute_single_action(input_dict=input_dict, explore=False, state=state)
            obs, reward, done, _, info = env.step(action)
            episode_stat["rewards"] = [reward] if "rewards" not in episode_stat else episode_stat["rewards"] + [reward]
            episode_stat["action_infos"] = [action_info] if "action_infos" not in episode_stat else episode_stat[
                                                                                                        "action_infos"] + [
                                                                                                        action_info]
            if env.rendering:
                env.render()
                time.sleep(0.05)

        episode_stat["states"] += [obs]
        episode_stat["actions"] += [action]
        episode_stat["infos"] += [info]
        # episode_stat["adaptor_gt"] = algo.model.log_dict_for_adaptor["ground_truth"] if use_robust_rl else None
        # episode_stat["adaptor_pred"] = algo.model.log_dict_for_adaptor["predicted"] if use_robust_rl else None

        trange_.set_description(f"Episode: {episode} Reward: {sum(episode_stat['rewards'])}")
        stat_dict[episode] = episode_stat
    return stat_dict


def limit_angles(angle):
    """
    Limit angles to [-pi, pi]
    """
    if np.pi < angle:
        angle = angle - 2 * np.pi
    if -np.pi > angle:
        angle = angle + 2 * np.pi

    return angle


def norm_angles_old(angle):
    if angle < 0:
        angle = (np.pi + abs(angle)) / (2 * np.pi)
    else:
        angle = angle / (2 * np.pi)

    return angle


def plot_agent_values(agent_dict):
    from matplotlib import pyplot as plt

    for name, agent in agent_dict.items():
        plt.plot(agent[:, 0], agent[:, 1], label=f"{name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Robot Position")
    plt.legend()
    plt.show()

    for name, agent in agent_dict.items():
        speed = np.sqrt(agent[:, 5] ** 2 + agent[:, 6] ** 2)
        plt.plot(speed, label=f"{name}")
    plt.title("Robot Speed")
    plt.xlabel("t")
    plt.ylabel("Speed")
    plt.legend()
    plt.show()

    for name, agent in agent_dict.items():
        plt.plot(agent[:, 2], label=f"{name}")
    plt.title("Theta(t)")
    plt.xlabel("t")
    plt.ylabel("Theta")
    plt.legend()
    plt.show()

    for name, agent in agent_dict.items():
        plt.plot(agent[:, 7], label=f"{name}")
    plt.title("Omega(t)")
    plt.xlabel("t")
    plt.ylabel("Omega")
    plt.legend()
    plt.show()


def get_env_data(dict_, key):
    if isinstance(dict_, dict):
        for k, v in dict_.items():
            if k == key:
                return v
            else:
                return get_env_data(v, key)
    else:
        return dict_


def create_lognormal_dist(mean_value=None, std_value=None, min_value=None, max_value=None, size=1, np_random=None):
    if mean_value is not None and std_value is not None:
        # Calculate the logarithms of the minimum and maximum values using the mean and standard deviation
        log_std_value = np.sqrt(np.log(1 + (std_value / mean_value) ** 2))
        log_min_value = np.log(mean_value) - 3 * log_std_value
        log_max_value = np.log(mean_value) + 3 * log_std_value
        min_value = np.exp(log_min_value)
        max_value = np.exp(log_max_value)
    elif min_value is not None and max_value is not None:
        pass
    else:
        raise ValueError("The mean and standard deviation or the minimum and maximum values must be provided.")

    # Generate sample from a log-uniform distribution between the logarithms of the minimum and maximum values
    log_uniform_samples = loguniform.rvs(a=min_value, b=max_value, size=size)

    # Transform the log-uniform samples to samples between the original minimum and maximum values
    return float(log_uniform_samples)


def create_uniform_dist(mean_value=None, std_value=None, min_value=None, max_value=None, size=1, np_random=None):
    return float(np_random.uniform(min_value, max_value, size=size))


def generate_random_value(default_value, min_value, max_value, ratio, np_random=None):
    if np_random is None:
        np_random = np.random
    log_min = np.log(min_value)
    log_max = np.log(max_value)
    log_range = log_max - log_min

    if ratio == 0.:
        return default_value
    elif ratio == 1.:
        log_value = np_random.uniform(0, log_range) + log_min
    else:
        log_deviation = log_range * (2 * ratio - 1) / 2
        log_deviation *= np_random.uniform(-1, 1)
        log_value = np.log(default_value) + log_deviation
    return np.exp(log_value)


def train_algo(algo, exit_criteria="episode_reward_mean", exit_treshold=0.95, max_iterations=None, verbose=False):
    best_checkpoint = None
    max_reward = - np.inf
    current_iteration = 0
    result = {"evaluation": {exit_criteria: -1.0}}
    current_checkpoint = None
    while result["evaluation"].get(exit_criteria, 0.) < exit_treshold:
        result.update(algo.train())
        current_iteration += 1
        if verbose:
            print(f"Iteration: {result['training_iteration']} Reward: {result[exit_criteria]},"
                  f" evaluation {exit_criteria}: {result['evaluation'].get(exit_criteria, 0.)} ")
        if result.get("evaluation", {}).get(exit_criteria, -1.) > max_reward:
            if best_checkpoint is not None:
                algo.delete_checkpoint(best_checkpoint)
            best_checkpoint = algo.save()
            max_reward = result.get("evaluation", {}).get(exit_criteria, 0.)
        else:
            if current_checkpoint is not None:
                algo.delete_checkpoint(current_checkpoint)
            current_checkpoint = algo.save()

        if max_iterations is not None and current_iteration >= max_iterations:
            print(f"Maximum number of iterations reached: {max_iterations}")
            break
    checkpoint_dir = algo.save()
    print(f"Checkpoint saved in directory {checkpoint_dir}")
    return algo


def custom_log_creator(custom_path, custom_str):
    from ray.tune.logger import UnifiedLogger

    timestr = datetime.today().strftime("%m%d_%H%M_")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def create_naming_convention(args):
    training_name = ""
    if args.use_robust_rl:
        training_name += "Robust_"
        if args.use_ae:
            training_name += "AE_"
    else:
        training_name += f"Base_{'LSTM_' if args.use_lstm else ''}"
    training_name += f"{args.model}_"
    training_name += f"{args.env[:3]}_"
    if args.random_range > 0.:
        training_name += f"rnd{str(args.random_range).replace('.', '')}_"
    if args.action_space_type == "continuous":
        training_name += "cont_"
    elif args.action_space_type == "discrete_simple":
        training_name += "simp_"
    elif args.action_space_type == "discrete_complex":
        training_name += "comp_"
    training_name += ''.join(i[0] for i in args.obs_space_type.split('_'))
    training_name += f"_{args.comment}"

    return training_name


def generate_cone_smartly(y, alpha, min_dist, np_random, x=0):
    y_rand = np_random.uniform(1.2 * min_dist, y - 1.2 * min_dist)
    r_rand = np_random.uniform(min_dist, y / 2)
    x_rand = 0
    m = np.cos(alpha) / np.sin(alpha)
    # handle the case when m is infinite or zero
    if m == 0 or np.isinf(m) or np.isnan(m):
        x_1 = x_rand + r_rand
        y_1 = y_rand
        x_2 = x_rand - r_rand
        y_2 = y_rand
    else:
        a = m ** 2 + 1
        b = -2 * y_rand * m ** 2 - 2 * y_rand
        c = -m ** 2 * r_rand ** 2 + m ** 2 * y_rand ** 2 + y_rand ** 2
        y_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        y_2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        x_1 = (y_1 - y_rand) / m
        x_2 = (y_2 - y_rand) / m
    # assert that no variable is nan or NaN
    assert not np.isnan(x_rand), f"x_rand is nan: {x_rand}"
    assert not np.isnan(y_rand), f"y_rand is nan: {y_rand}"
    assert not np.isnan(x_1), f"x_1 is nan: {x_1}"
    assert not np.isnan(y_1), f"y_1 is nan: {y_1}"
    assert not np.isnan(x_2), f"x_2 is nan: {x_2}"
    assert not np.isnan(y_2), f"y_2 is nan: {y_2}"

    return [(x_rand, y_rand), (x_1, y_1), (x_2, y_2)]


def calculate_dist_of_points(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
