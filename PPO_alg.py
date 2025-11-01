import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import yfinance as yf
from datetime import datetime, timedelta

# Technical Indicators
class TechnicalIndicators:
    @staticmethod
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.values, signal_line.values
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band.values, sma.values, lower_band.values
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().values
        return atr
    
    @staticmethod
    def calculate_obv(close, volume):
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return obv


# Enhanced Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 128]):
        super(ActorCritic, self).__init__()
        
        # Deep shared feature extractor with LayerNorm and Dropout
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, state_value
    
    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_value, dist_entropy


# PPO Memory Buffer
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_values.clear()


# Enhanced Trading Environment with Real Market Data
class EnhancedTradingEnvironment:
    def __init__(self, ticker='SPY', start_date='2020-01-01', end_date='2024-01-01',
                 initial_balance=100000, transaction_cost=0.001, window_size=50):
        self.ticker = ticker
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.window_size = window_size
        
        # Download real market data
        print(f"Downloading {ticker} data from {start_date} to {end_date}...")
        self.data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if len(self.data) == 0:
            raise ValueError("No data downloaded. Check ticker and date range.")
        
        self._prepare_features()
        self.reset()
    
    def _prepare_features(self):
        # Calculate technical indicators
        close = self.data['Close'].values
        high = self.data['High'].values
        low = self.data['Low'].values
        volume = self.data['Volume'].values
        
        # RSI
        rsi = TechnicalIndicators.calculate_rsi(close, 14)
        
        # MACD
        macd, signal = TechnicalIndicators.calculate_macd(close)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(close)
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_position = (close - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        atr = TechnicalIndicators.calculate_atr(high, low, close)
        
        # OBV
        obv = TechnicalIndicators.calculate_obv(close, volume)
        
        # Moving Averages
        sma_20 = pd.Series(close).rolling(window=20).mean().values
        sma_50 = pd.Series(close).rolling(window=50).mean().values
        ema_12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        
        # Price momentum
        returns_1d = np.diff(close, prepend=close[0]) / close
        returns_5d = (close - np.roll(close, 5)) / np.roll(close, 5)
        returns_20d = (close - np.roll(close, 20)) / np.roll(close, 20)
        
        # Volatility
        volatility = pd.Series(returns_1d).rolling(window=20).std().values
        
        # Volume indicators
        volume_sma = pd.Series(volume).rolling(window=20).mean().values
        volume_ratio = volume / (volume_sma + 1e-8)
        
        # Create feature DataFrame
        self.features = pd.DataFrame({
            'close': close,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'bb_width': bb_width,
            'bb_position': bb_position,
            'atr': atr / close,  # normalized
            'obv': obv / np.max(np.abs(obv)),  # normalized
            'sma_20': sma_20 / close,
            'sma_50': sma_50 / close,
            'ema_12': ema_12 / close,
            'returns_1d': returns_1d,
            'returns_5d': returns_5d,
            'returns_20d': returns_20d,
            'volatility': volatility,
            'volume_ratio': volume_ratio
        })
        
        # Fill NaN values
        self.features = self.features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"Prepared {len(self.features)} trading days with {len(self.features.columns)} features")
    
    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = 0
        self.winning_trades = 0
        
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.features):
            return np.zeros(self.window_size * len(self.features.columns) + 4)
        
        # Get window of features
        window_features = self.features.iloc[self.current_step - self.window_size:self.current_step]
        
        # Flatten window features
        state = window_features.values.flatten()
        
        # Add account state
        current_price = self.features.iloc[self.current_step]['close']
        position_value = self.shares * current_price
        self.net_worth = self.balance + position_value
        
        account_state = np.array([
            self.shares / 100,  # normalized position
            self.balance / self.initial_balance,
            self.net_worth / self.initial_balance,
            position_value / (self.net_worth + 1e-8)
        ])
        
        return np.concatenate([state, account_state])
    
    def step(self, action):
        # Actions: 0=hold, 1=buy, 2=sell
        if self.current_step >= len(self.features) - 1:
            return self._get_state(), 0, True
        
        current_price = self.features.iloc[self.current_step]['close']
        prev_net_worth = self.net_worth
        
        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                shares_to_buy = (self.balance * 0.95) / current_price  # Use 95% of balance
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.shares += shares_to_buy
                    self.balance -= cost
                    self.trades += 1
        
        elif action == 2:  # Sell
            if self.shares > 0:
                proceeds = self.shares * current_price * (1 - self.transaction_cost)
                if proceeds > self.shares * current_price * (1 + self.transaction_cost):
                    self.winning_trades += 1
                self.balance += proceeds
                self.shares = 0
                self.trades += 1
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        next_price = self.features.iloc[self.current_step]['close']
        position_value = self.shares * next_price
        self.net_worth = self.balance + position_value
        
        # Reward is based on net worth change
        reward = (self.net_worth - prev_net_worth) / prev_net_worth * 100
        
        # Bonus for beating market
        market_return = (next_price - current_price) / current_price * 100
        reward += (reward - market_return) * 0.1
        
        # Penalty for drawdown
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        reward -= drawdown * 10
        
        done = self.current_step >= len(self.features) - 1
        
        return self._get_state(), reward, done


# Enhanced PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.2, 
                 K_epochs=10, gae_lambda=0.95, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.state_values.append(state_value)
        
        return action
    
    def compute_gae(self, rewards, values, is_terminals):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - is_terminals[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - is_terminals[t]) * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, memory):
        # Convert to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.tensor(memory.actions, dtype=torch.long).detach()
        old_logprobs = torch.stack(memory.logprobs, dim=0).detach()
        
        # Compute advantages using GAE
        values = [v.item() for v in memory.state_values]
        advantages = self.compute_gae(memory.rewards, values, memory.is_terminals)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + torch.tensor(values, dtype=torch.float32)
        
        # Optimize policy for K epochs
        total_loss = 0
        for _ in range(self.K_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # Final loss with value function and entropy bonuses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = self.MseLoss(state_values, returns)
            entropy_loss = -dist_entropy.mean()
            
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Copy new weights to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        return total_loss / self.K_epochs
    
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
    
    def load(self, filepath):
        self.policy.load_state_dict(torch.load(filepath))
        self.policy_old.load_state_dict(torch.load(filepath))


# Training function
def train_ppo(ticker='SPY', episodes=500, update_timestep=500):
    # Create environment with real market data
    env = EnhancedTradingEnvironment(
        ticker=ticker,
        start_date='2018-01-01',
        end_date='2023-12-31',
        initial_balance=100000
    )
    
    state_dim = len(env._get_state())
    action_dim = 3  # hold, buy, sell
    
    agent = PPOAgent(
        state_dim, 
        action_dim,
        lr=5e-5,
        gamma=0.99,
        eps_clip=0.2,
        K_epochs=10,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_coef=0.5
    )
    
    memory = PPOMemory()
    
    print("\n" + "="*70)
    print(f"Training PPO Agent on {ticker} - Real Market Data")
    print("="*70)
    print(f"State Dimension: {state_dim}")
    print(f"Episodes: {episodes}, Update every {update_timestep} steps")
    print("-"*70)
    
    time_step = 0
    episode_rewards = []
    episode_returns = []
    best_return = -float('inf')
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        while True:
            time_step += 1
            
            # Select action
            action = agent.select_action(state, memory)
            
            # Take step
            next_state, reward, done = env.step(action)
            
            # Store in memory
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            episode_reward += reward
            state = next_state
            
            # Update policy
            if time_step % update_timestep == 0:
                loss = agent.update(memory)
                memory.clear()
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        portfolio_return = (env.net_worth - env.initial_balance) / env.initial_balance * 100
        episode_returns.append(portfolio_return)
        
        # Track best model
        if portfolio_return > best_return:
            best_return = portfolio_return
            agent.save(f"best_ppo_{ticker}.pth")
        
        if episode % 25 == 0:
            avg_reward = np.mean(episode_rewards[-25:])
            avg_return = np.mean(episode_returns[-25:])
            win_rate = env.winning_trades / env.trades if env.trades > 0 else 0
            
            print(f"Ep {episode:4d} | Reward: {avg_reward:7.2f} | Return: {avg_return:6.2f}% | "
                  f"Net Worth: ${env.net_worth:,.0f} | Trades: {env.trades:3d} | "
                  f"Win Rate: {win_rate:.2%}")
    
    print("-"*70)
    print("Training complete!")
    print(f"Best Return: {best_return:.2f}%")
    print(f"Final Return: {portfolio_return:.2f}%")
    print(f"Total Trades: {env.trades}")
    print("="*70)
    
    return agent, episode_rewards, episode_returns


# Backtesting function
def backtest_agent(agent, ticker='SPY', start_date='2023-01-01', end_date='2024-12-31'):
    """Test trained agent on unseen data"""
    print("\n" + "="*70)
    print(f"BACKTESTING on {ticker}: {start_date} to {end_date}")
    print("="*70)
    
    env = EnhancedTradingEnvironment(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_balance=100000
    )
    
    state = env.reset()
    done = False
    trades_log = []
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = agent.policy(state_tensor)
            action = torch.argmax(action_probs).item()
        
        current_price = env.features.iloc[env.current_step]['close']
        state, reward, done = env.step(action)
        
        if action != 0:  # Log trades
            trades_log.append({
                'step': env.current_step,
                'action': ['HOLD', 'BUY', 'SELL'][action],
                'price': current_price,
                'shares': env.shares,
                'balance': env.balance,
                'net_worth': env.net_worth
            })
    
    # Calculate metrics
    total_return = (env.net_worth - env.initial_balance) / env.initial_balance * 100
    
    # Buy and hold comparison
    buy_hold_return = (env.features.iloc[-1]['close'] - env.features.iloc[env.window_size]['close']) / env.features.iloc[env.window_size]['close'] * 100
    
    print("-"*70)
    print("BACKTEST RESULTS:")
    print(f"Initial Balance: ${env.initial_balance:,.2f}")
    print(f"Final Net Worth: ${env.net_worth:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Alpha (vs Buy & Hold): {total_return - buy_hold_return:.2f}%")
    print(f"Total Trades: {env.trades}")
    print(f"Winning Trades: {env.winning_trades}")
    print(f"Win Rate: {env.winning_trades/env.trades*100 if env.trades > 0 else 0:.2f}%")
    print(f"Max Drawdown: {(env.max_net_worth - env.net_worth) / env.max_net_worth * 100:.2f}%")
    print("="*70)
    
    return trades_log, total_return


# Run training
if __name__ == "__main__":
    # Example 1: Train on SPY (S&P 500 ETF)
    print("\n🚀 TRAINING PPO AGENT ON REAL MARKET DATA")
    agent, rewards, returns = train_ppo(ticker='SPY', episodes=300, update_timestep=500)
    
    print("\n📊 TRAINING RESULTS:")
    print(f"Average Return (last 50 episodes): {np.mean(returns[-50:]):.2f}%")
    print(f"Max Return: {np.max(returns):.2f}%")
    print(f"Min Return: {np.min(returns):.2f}%")
    print(f"Std Dev: {np.std(returns[-50:]):.2f}%")
    
    # Backtest on unseen data (2024)
    trades, test_return = backtest_agent(agent, ticker='SPY', start_date='2024-01-01', end_date='2024-10-31')
    
    # Save final model
    agent.save("ppo_trading_agent_final.pth")
    print("\n✅ Model saved as 'ppo_trading_agent_final.pth'")
    
    print("\n" + "="*70)
    print("EXPERIMENT WITH DIFFERENT TICKERS:")
    print("- Change ticker='SPY' to 'AAPL', 'TSLA', 'NVDA', 'MSFT', etc.")
    print("- Adjust date ranges for different market conditions")
    print("- Modify hyperparameters in PPOAgent() for optimization")
    print("="*70)
