import ray
from algorithms.sac import SAC


if __name__ == '__main__':
    mode = 'train'  # 'train' or 'eval
    log_env = True
    rendering = False

    ray.init(log_to_driver=True)

    if mode == 'train':
        sac = SAC()
        sac.logger.log(sac.logger.parameter)
        sac.run()

    elif mode == 'eval':
        path_to_load = '/Users/fpeti/PycharmProjects/ESCP/saved_model/ESCP_DynDiffRobotESCP-v0-use_rmdm-rnn_len_16-bottle_neck-stop_pg_for_ep-ep_dim_2-19_1000tasksSeed19GGOR'
        sac = SAC(path_to_load=path_to_load, log_env=log_env)
        sac.logger.log(sac.logger.parameter)
        # NOTE: if rendering is True, it should be set to true in the env_config of the DDR env
        sac.eval(episodes=200, rendering=rendering)
