QuantAI: Advanced Algorithmic Trading Framework

A modern, modular platform for deep reinforcement learning and machine learning–based trading strategies.

🌍 Overview

QuantAI is a full-stack research and execution framework for algorithmic trading.
It integrates Deep Reinforcement Learning (DRL) and Ensemble Machine Learning (EML) to model both discrete and continuous trading decisions.

QuantAI is designed to let quants, researchers, and data scientists develop, backtest, and deploy trading systems with ease, rigor, and reproducibility.

🧠 Core Objectives

Build and benchmark intelligent trading agents (RL + ML).

Compare algorithmic methods across multiple market regimes.

Provide standardized evaluation metrics (Sharpe, Sortino, Max Drawdown).

Enable robust, configurable experimentation pipelines.

🧩 Repository Structure
QuantAI/
├── algorithms/
│   ├── drl/
│   │   ├── dqn_agent.py        # Deep Q-Network (discrete actions)
│   │   ├── ddpg_agent.py       # Deep Deterministic Policy Gradient (continuous actions)
│   │   └── ppo_agent.py        # Proximal Policy Optimization (robust policy gradient)
│   └── ml/
│       ├── random_forest.py    # Random Forest ensemble model
│       └── xgboost_model.py    # eXtreme Gradient Boosting model
│
├── data/
│   ├── historical_data/        # Historical OHLCV datasets
│   └── processing_scripts/     # Data cleaning and feature engineering tools
│
├── backtester/
│   ├── backtest_engine.py      # Core simulation and execution engine
│   └── reporting.py            # Performance metrics and visualization utilities
│
├── config/
│   └── settings.json           # Environment and hyperparameter configuration
│
├── notebooks/
│   └── Exploratory_Analysis.ipynb
│
├── requirements.txt
└── README.md

⚙️ Installation
Prerequisites

Python ≥ 3.9

pip (Python package installer)

Setup
# Clone the repository
git clone https://github.com/YourUsername/QuantAI.git
cd QuantAI

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

🧠 Algorithm Modules
Deep Reinforcement Learning (DRL)
Algorithm	Type	Description
DQN	Value-Based	Combines Q-Learning with deep networks for discrete trading (Buy/Sell/Hold).
DDPG	Actor–Critic	Learns continuous control policies for position sizing or allocation.
PPO	Actor–Critic	On-policy, clipped objective for stable and efficient policy optimization.
Ensemble Machine Learning (EML)
Algorithm	Type	Description
Random Forest	Bagging Ensemble	Aggregates multiple decision trees to reduce overfitting and improve generalization.
XGBoost	Boosting Ensemble	Sequentially corrects model errors using gradient boosting for predictive precision.
🚀 Usage Guide
1. Prepare Data

Place cleaned historical OHLCV files into:

data/historical_data/


Use scripts in data/processing_scripts/ for preprocessing and feature engineering.

2. Configure Settings

Edit config/settings.json to define:

Algorithm type (dqn, ddpg, ppo, random_forest, xgboost)

Asset symbol, time frame

Hyperparameters (learning rate, batch size, etc.)

3. Run Backtests
python backtester/backtest_engine.py --config config/settings.json

4. View Reports

Generated metrics and plots are saved in the output directory:

Equity Curve

Return Distribution

Sharpe & Sortino Ratios

Maximum Drawdown

Use backtester/reporting.py for comparative analysis across multiple strategies.

📊 Example Workflow
# Run PPO on BTC/USD with a 1-hour timeframe
python backtester/backtest_engine.py --config config/ppo_btcusd.json

# Compare Random Forest vs DQN
python backtester/reporting.py --compare rf_dqn_results/

🤝 Contributing

Contributions are encouraged.

Fork the repository

Create a feature branch

Commit and push your changes

Submit a pull request

Please ensure your contributions follow clean coding, documentation, and reproducibility standards.

🧾 License

Distributed under the MIT License.
See LICENSE for full terms.

📚 Future Extensions

Integration with live trading APIs (Binance, Alpaca, Interactive Brokers).

Portfolio-level RL with multi-asset optimization.

Federated or online learning capabilities.

Real-time monitoring dashboard using Streamlit.
