import gymnasium as gym
import gym_trading_env
import numpy as np

def make_trading_env(df):
    env = gym.make("TradingEnv",
            name= "stock",
            df = df, # Your dataset with your custom features
            positions = np.linspace(0,1,11).tolist(),
            portfolio_initial_value = 1000,
            trading_fees = 0.03,
            initial_position = 0,
            verbose=1
        )
    return env