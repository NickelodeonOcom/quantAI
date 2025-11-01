import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output in [-1, 1]
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.001, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer and noise
        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)
        
    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

# Data loader with technical indicators
def load_market_data(ticker='AAPL', start='2020-01-01', end='2024-01-01'):
    """Download and prepare market data with technical indicators"""
    print(f"Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")
    
    # Calculate technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'], periods=14)
    df['MACD'], df['Signal'] = calculate_macd(df['Close'])
    df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['Close'])
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Drop NaN values
    df = df.dropna()
    
    print(f"Loaded {len(df)} trading days")
    return df

def calculate_rsi(prices, periods=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, lower_band

# Trading Environment with real market data
class TradingEnvironment:
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, 
                 lookback_window=20):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Prepare features
        self.feature_columns = ['Returns', 'MA_5', 'MA_20', 'MA_50', 'RSI', 
                               'MACD', 'Signal', 'BB_upper', 'BB_lower', 'Volume_Ratio']
        
        # Normalize features
        self.scaler = StandardScaler()
        self.df[self.feature_columns] = self.scaler.fit_transform(
            self.df[self.feature_columns]
        )
        
        self.reset()
        
    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # Current position (-1 to 1)
        self.shares_held = 0
        self.total_asset_value = self.initial_balance
        self.trades = []
        self.portfolio_values = [self.initial_balance]
        
        return self._get_state()
    
    def _get_state(self):
        """Get current market state with technical indicators"""
        # Get lookback window of features
        window_data = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        features = []
        for col in self.feature_columns:
            features.extend(window_data[col].values)
        
        # Add current position and portfolio info
        current_price = self.df.iloc[self.current_step]['Close']
        features.extend([
            self.position,
            self.balance / self.initial_balance,
            self.total_asset_value / self.initial_balance,
            (current_price * self.shares_held) / self.initial_balance if self.shares_held > 0 else 0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action):
        """Execute trading action"""
        current_price = self.df.iloc[self.current_step]['Close']
        action_value = action[0]  # Position target in [-1, 1]
        
        # Calculate desired position change
        position_change = action_value - self.position
        
        # Calculate number of shares to trade
        max_shares = self.total_asset_value / current_price
        shares_to_trade = position_change * max_shares
        
        # Execute trade with transaction costs
        trade_value = abs(shares_to_trade * current_price)
        trade_cost = trade_value * self.transaction_cost
        
        if shares_to_trade > 0:  # Buying
            if trade_value + trade_cost <= self.balance:
                self.shares_held += shares_to_trade
                self.balance -= (trade_value + trade_cost)
                self.position = action_value
                self.trades.append(('BUY', shares_to_trade, current_price))
        elif shares_to_trade < 0:  # Selling
            shares_to_sell = min(abs(shares_to_trade), self.shares_held)
            self.shares_held -= shares_to_sell
            self.balance += (shares_to_sell * current_price - trade_cost)
            self.position = (self.shares_held * current_price) / self.total_asset_value if self.total_asset_value > 0 else 0
            self.trades.append(('SELL', shares_to_sell, current_price))
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate portfolio value
        if not done:
            next_price = self.df.iloc[self.current_step]['Close']
        else:
            next_price = current_price
            
        old_total_value = self.total_asset_value
        self.total_asset_value = self.balance + self.shares_held * next_price
        self.portfolio_values.append(self.total_asset_value)
        
        # Calculate reward as portfolio return
        reward = (self.total_asset_value - old_total_value) / old_total_value
        
        # Add penalty for excessive trading
        if len(self.trades) > 0:
            reward -= 0.0001  # Small penalty for each trade
        
        next_state = self._get_state() if not done else self._get_state()
        
        return next_state, reward, done, {
            'portfolio_value': self.total_asset_value,
            'position': self.position,
            'balance': self.balance
        }
    
    def get_state_dim(self):
        return len(self._get_state())
    
    def get_portfolio_stats(self):
        """Calculate portfolio performance statistics"""
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        total_return = (self.total_asset_value - self.initial_balance) / self.initial_balance
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'final_value': self.total_asset_value,
            'num_trades': len(self.trades)
        }
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        peak = self.portfolio_values[0]
        max_dd = 0
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

# Training function
def train_ddpg(agent, env, episodes=100):
    """Train DDPG agent on market data"""
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        agent.noise.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, add_noise=True)
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            actor_loss, critic_loss = agent.train()
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        rewards_history.append(episode_reward)
        stats = env.get_portfolio_stats()
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Total Return: {stats['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
            print(f"  Final Value: ${stats['final_value']:.2f}")
            print(f"  Num Trades: {stats['num_trades']}")
            print()
    
    return rewards_history

# Main execution
if __name__ == "__main__":
    # Load market data
    ticker = 'AAPL'  # Change to any ticker
    df = load_market_data(ticker=ticker, start='2020-01-01', end='2024-01-01')
    
    # Create environment and agent
    env = TradingEnvironment(df, initial_balance=10000)
    state_dim = env.get_state_dim()
    action_dim = 1
    
    print(f"\nState dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Training data: {len(df)} days\n")
    
    agent = DDPGAgent(state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3)
    
    # Train the agent
    print("=" * 50)
    print("TRAINING DDPG AGENT")
    print("=" * 50)
    rewards = train_ddpg(agent, env, episodes=100)
    
    # Test the trained agent
    print("\n" + "=" * 50)
    print("TESTING TRAINED AGENT")
    print("=" * 50)
    state = env.reset()
    
    while True:
        action = agent.select_action(state, add_noise=False)
        state, reward, done, info = env.step(action)
        if done:
            break
    
    stats = env.get_portfolio_stats()
    print(f"\nFinal Test Results:")
    print(f"  Total Return: {stats['total_return']:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"  Final Portfolio Value: ${stats['final_value']:.2f}")
    print(f"  Number of Trades: {stats['num_trades']}")
    
    # Calculate buy and hold benchmark
    buy_hold_return = ((df.iloc[-1]['Close'] - df.iloc[env.lookback_window]['Close']) / 
                       df.iloc[env.lookback_window]['Close']) * 100
    print(f"\nBuy & Hold Benchmark: {buy_hold_return:.2f}%")
    print(f"Agent vs Benchmark: {stats['total_return'] - buy_hold_return:+.2f}%")
