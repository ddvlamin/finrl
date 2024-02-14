from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, simulator, features, window_size=4*86400, render_mode=None):
        """

        :param simulator:
        :param features:
        :param window_size: window size in seconds (default value: four days)
        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode

        self._simulator = simulator
        self._features = features
        self._window_size = window_size

        # episode
        self._truncated = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        # spaces
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(simulator.npositions,), dtype=np.float32,
        )
        #INF = 1e10
        #self.observation_space = gym.spaces.Box(
        #    low=-INF, high=INF, shape=self.shape, dtype=np.float32,
        #)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._simulator.reset()
        self._initial_value = self._simulator.total_value

        self.action_space.seed(int((self.np_random.uniform(0, seed if seed is not None else 1))))

        self._truncated = False
        self._total_reward = 0.
        self._total_profit = 0.  # unit

        self._first_rendering = True
        self.history = {}

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def step(self, position):
        self._simulator.reposition(position)
        trade_value = self._simulator.total_value

        self._truncated = self._simulator.step()
        new_value = self._simulator.total_value

        step_reward = (new_value - trade_value) / trade_value
        self._total_reward += step_reward

        self._total_profit = new_value - self._initial_value #should be more accurate than cumulating step diffs

        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == 'human':
            self._render_frame()

        if self._truncated:
            self.reset()

        return observation, step_reward, False, self._truncated, info

    def _get_info(self):
        return dict(
            total_reward=self._total_reward,
            total_profit=self._total_profit,
            positions=self._simulator.current_positions
        )

    def _get_observation(self):
        end_date = self._simulator.current_tick
        start_date = end_date - timedelta(seconds=self.window_size)
        window = self._features.loc[start_date:end_date]
        return window.to_numpy().reshape((1,window.shape[0]*window.shape[1]))

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):
        pass
    def render_all(self, title=None):
        pass

    def close(self):
        plt.close()

    def save_rendering(self, filepath):
        plt.savefig(filepath)

    def pause_rendering(self):
        plt.show()
