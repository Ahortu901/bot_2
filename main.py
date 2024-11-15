import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C
import alpaca_trade_api as tradeapi
import yfinance as yf
import matplotlib.pyplot as plt

# Set your Alpaca API credentials
ALPACA_API_KEY = 'PKQC5VJBWBPBI8TRAR18'
ALPACA_API_SECRET = 'kpTpgfrggF1KkoppSM137EtKHKD0dNXA89H8uIbj'
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # For paper trading

# Alpaca API setup
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL, api_version='v2')

# Enhanced Trading Environment with Alpaca Integration
class MultiAssetTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, alpaca_api=None):
        super(MultiAssetTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.current_step = 0
        
        # Balance and portfolio state
        self.balance = initial_balance
        self.holdings = np.zeros(len(self.data.columns) - 1)  # Holdings for each asset
        self.net_worth = initial_balance
        
        # Action space: [-1, 1] for each asset (proportion of balance to invest)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.holdings),), dtype=np.float32)
        
        # Observation space: prices of all assets, balance, and holdings
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.holdings) + len(self.data.columns) - 1 + 1,), dtype=np.float32
        )

        # Set Alpaca API client
        self.api = alpaca_api
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(len(self.data.columns) - 1)
        self.net_worth = self.initial_balance
        return self._get_observation()
    
    def _get_observation(self):
        prices = self.data.iloc[self.current_step, 1:].values  # Current prices for all assets
        return np.concatenate(([self.balance], self.holdings, prices), axis=0)
    
    def step(self, actions):
        prices = self.data.iloc[self.current_step, 1:].values
        prev_net_worth = self.net_worth
        
        # Execute trades through Alpaca
        for i, action in enumerate(actions):
            allocation = self.balance * action  # Amount to trade
            if allocation > 0:  # Buy
                cost = allocation + (allocation * self.transaction_cost)
                if cost <= self.balance:
                    self.holdings[i] += allocation / prices[i]
                    self.balance -= cost
                    # Place buy order using Alpaca
                    self.place_order('buy', self.data.columns[i], allocation / prices[i])
            elif allocation < 0:  # Sell
                sell_amount = min(-allocation, self.holdings[i] * prices[i])
                revenue = sell_amount - (sell_amount * self.transaction_cost)
                self.holdings[i] -= sell_amount / prices[i]
                self.balance += revenue
                # Place sell order using Alpaca
                self.place_order('sell', self.data.columns[i], sell_amount / prices[i])
        
        # Update net worth
        self.net_worth = self.balance + np.sum(self.holdings * prices)
        reward = self.net_worth - prev_net_worth
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}

    def place_order(self, side, symbol, qty):
        try:
            if side == 'buy':
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            elif side == 'sell':
                api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            print(f"Order placed: {side} {qty} of {symbol}")
        except Exception as e:
            print(f"Error placing order: {e}")

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}, Net Worth: {self.net_worth}")

# Fetch Data for Multiple Assets
def fetch_multiple_assets(tickers=["GC=F", "SI=F", "^GSPC"], start_date="2010-01-01", end_date="2023-12-31"):
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)  # Ensure no NaN values remain
    data.reset_index(drop=True, inplace=True)
    return data

# Train the Model Using PPO or A2C
def train_rl_model(data, algorithm="PPO", timesteps=10000):
    env = MultiAssetTradingEnv(data, alpaca_api=api)
    model_class = PPO if algorithm == "PPO" else A2C
    model = model_class("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model, env

# Evaluate the Model
def evaluate_model(model, env, data):
    obs = env.reset()
    net_worths = []
    
    for _ in range(len(data) - 1):
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        net_worths.append(env.net_worth)
        if done:
            break
    
    # Plot net worth over time
    plt.plot(net_worths)
    plt.title("Net Worth Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Net Worth")
    plt.show()
    return net_worths

# Portfolio Metrics
def calculate_portfolio_metrics(net_worths):
    returns = np.diff(net_worths) / net_worths[:-1]
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    max_drawdown = (np.min(net_worths) - np.max(net_worths)) / np.max(net_worths)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}")

# Main Function
if __name__ == "__main__":
    # Fetch data for multiple assets
    data = fetch_multiple_assets()
    
    # Check fetched data shape
    print(f"Fetched data shape: {data.shape}")
    print(f"Data preview:\n{data.head()}")
    
    # Train model
    print("Training the model...")
    model, env = train_rl_model(data, algorithm="PPO", timesteps=10000)
    
    # Evaluate model
    print("Evaluating the model...")
    net_worths = evaluate_model(model, env, data)
    
    # Calculate portfolio metrics
    print("Calculating portfolio metrics...")
    calculate_portfolio_metrics(net_worths)
