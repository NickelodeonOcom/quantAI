import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class QuantXGBoostStrategy:
    """
    Full AI Quantitative Trading Strategy using XGBoost
    """
    
    def __init__(self, ticker='SPY', start_date='2015-01-01', end_date='2024-12-31'):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def fetch_data(self):
        """Fetch historical price data"""
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Downloaded {len(self.data)} rows of data")
        return self.data
    
    def create_technical_features(self):
        """Create comprehensive technical indicators as features"""
        df = self.data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Moving average crossovers
        df['SMA_5_20_diff'] = df['SMA_5'] - df['SMA_20']
        df['SMA_20_50_diff'] = df['SMA_20'] - df['SMA_50']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Momentum indicators
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Volatility
        df['Volatility_10'] = df['Returns'].rolling(window=10).std()
        df['Volatility_30'] = df['Returns'].rolling(window=30).std()
        
        # Volume features
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # ATR (Average True Range)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
        
        # Lag features (previous days' returns)
        for lag in range(1, 6):
            df[f'Lag_{lag}_return'] = df['Returns'].shift(lag)
        
        self.data = df
        return df
    
    def create_target(self, forward_days=1, threshold=0.0):
        """
        Create binary target variable
        1 = price will go up by more than threshold in forward_days
        0 = otherwise
        """
        self.data['Future_Return'] = self.data['Close'].shift(-forward_days) / self.data['Close'] - 1
        self.data['Target'] = (self.data['Future_Return'] > threshold).astype(int)
        return self.data
    
    def prepare_train_test(self, train_split=0.8):
        """Prepare training and testing datasets"""
        # Remove rows with NaN values
        df_clean = self.data.dropna()
        
        # Select feature columns (exclude price/volume columns and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 
                       'Future_Return', 'Target', 'BB_middle', 'BB_std', 'BB_upper', 'BB_lower']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean['Target']
        
        # Time-based split (important for time series)
        split_idx = int(len(X) * train_split)
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        # Store test dates for backtesting
        self.test_dates = df_clean.index[split_idx:]
        self.test_prices = df_clean['Close'].iloc[split_idx:]
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
        print(f"Number of features: {len(feature_cols)}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_model(self, optimize=False):
        """Train XGBoost model with optional hyperparameter optimization"""
        print("\nTraining XGBoost model...")
        
        if optimize:
            print("Performing hyperparameter optimization...")
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            tscv = TimeSeriesSplit(n_splits=3)
            
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train_scaled, self.y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default good parameters
            self.model = xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.05,
                n_estimators=200,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            self.model.fit(
                self.X_train_scaled, 
                self.y_train,
                eval_set=[(self.X_test_scaled, self.y_test)],
                early_stopping_rounds=20,
                verbose=False
            )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training complete!")
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        
        # Prediction probabilities
        self.y_test_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        
        print(f"\nTraining Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")
        print(f"Testing Accuracy: {accuracy_score(self.y_test, y_test_pred):.4f}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(self.y_test, y_test_pred))
        
        return y_test_pred
    
    def backtest(self, initial_capital=100000, transaction_cost=0.001):
        """Backtest the trading strategy"""
        print("\n" + "="*50)
        print("BACKTESTING RESULTS")
        print("="*50)
        
        # Create backtesting dataframe
        backtest_df = pd.DataFrame({
            'Date': self.test_dates,
            'Price': self.test_prices,
            'Prediction': self.model.predict(self.X_test_scaled),
            'Probability': self.y_test_proba,
            'Actual': self.y_test
        })
        
        # Calculate strategy returns
        backtest_df['Returns'] = backtest_df['Price'].pct_change()
        
        # Strategy: Only take position when prediction is 1 and probability > 0.6
        backtest_df['Signal'] = ((backtest_df['Prediction'] == 1) & 
                                 (backtest_df['Probability'] > 0.6)).astype(int)
        
        # Apply transaction costs
        backtest_df['Position_Change'] = backtest_df['Signal'].diff().abs()
        backtest_df['Transaction_Cost'] = backtest_df['Position_Change'] * transaction_cost
        
        # Calculate strategy returns
        backtest_df['Strategy_Returns'] = (backtest_df['Signal'].shift(1) * 
                                           backtest_df['Returns'] - 
                                           backtest_df['Transaction_Cost'])
        
        # Calculate cumulative returns
        backtest_df['Cumulative_Market'] = (1 + backtest_df['Returns']).cumprod()
        backtest_df['Cumulative_Strategy'] = (1 + backtest_df['Strategy_Returns']).cumprod()
        
        # Performance metrics
        total_return_market = backtest_df['Cumulative_Market'].iloc[-1] - 1
        total_return_strategy = backtest_df['Cumulative_Strategy'].iloc[-1] - 1
        
        # Sharpe Ratio (annualized)
        sharpe_market = (backtest_df['Returns'].mean() / backtest_df['Returns'].std()) * np.sqrt(252)
        sharpe_strategy = (backtest_df['Strategy_Returns'].mean() / 
                          backtest_df['Strategy_Returns'].std()) * np.sqrt(252)
        
        # Max Drawdown
        cumulative_strategy = backtest_df['Cumulative_Strategy']
        running_max = cumulative_strategy.expanding().max()
        drawdown = (cumulative_strategy - running_max) / running_max
        max_drawdown = drawdown.min()
        
        print(f"\nInitial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital (Strategy): ${initial_capital * (1 + total_return_strategy):,.2f}")
        print(f"Final Capital (Buy & Hold): ${initial_capital * (1 + total_return_market):,.2f}")
        
        print(f"\nTotal Return (Strategy): {total_return_strategy*100:.2f}%")
        print(f"Total Return (Market): {total_return_market*100:.2f}%")
        print(f"Outperformance: {(total_return_strategy - total_return_market)*100:.2f}%")
        
        print(f"\nSharpe Ratio (Strategy): {sharpe_strategy:.2f}")
        print(f"Sharpe Ratio (Market): {sharpe_market:.2f}")
        
        print(f"\nMax Drawdown: {max_drawdown*100:.2f}%")
        
        print(f"\nNumber of Trades: {int(backtest_df['Position_Change'].sum())}")
        print(f"Win Rate: {(backtest_df[backtest_df['Signal'] == 1]['Returns'] > 0).mean()*100:.2f}%")
        
        self.backtest_df = backtest_df
        return backtest_df
    
    def plot_results(self):
        """Visualize results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'XGBoost Quant Trading Strategy - {self.ticker}', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        ax1.plot(self.backtest_df['Date'], self.backtest_df['Cumulative_Market'], 
                label='Buy & Hold', linewidth=2)
        ax1.plot(self.backtest_df['Date'], self.backtest_df['Cumulative_Strategy'], 
                label='XGBoost Strategy', linewidth=2)
        ax1.set_title('Cumulative Returns Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Importance
        ax2 = axes[0, 1]
        top_features = self.feature_importance.head(15)
        ax2.barh(range(len(top_features)), top_features['importance'])
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'])
        ax2.set_title('Top 15 Feature Importance')
        ax2.set_xlabel('Importance')
        ax2.invert_yaxis()
        
        # 3. Trading Signals
        ax3 = axes[1, 0]
        ax3.plot(self.backtest_df['Date'], self.backtest_df['Price'], 
                label='Price', linewidth=1, alpha=0.7)
        buy_signals = self.backtest_df[self.backtest_df['Signal'] == 1]
        ax3.scatter(buy_signals['Date'], buy_signals['Price'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        ax3.set_title('Trading Signals')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Prediction Probability Distribution
        ax4 = axes[1, 1]
        ax4.hist(self.backtest_df['Probability'], bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0.6, color='r', linestyle='--', linewidth=2, label='Threshold (0.6)')
        ax4.set_title('Prediction Probability Distribution')
        ax4.set_xlabel('Probability')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Drawdown
        ax5 = axes[2, 0]
        cumulative = self.backtest_df['Cumulative_Strategy']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax5.fill_between(self.backtest_df['Date'], drawdown * 100, 0, alpha=0.5, color='red')
        ax5.set_title('Strategy Drawdown')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Drawdown (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Monthly Returns Heatmap
        ax6 = axes[2, 1]
        monthly_returns = self.backtest_df.set_index('Date')['Strategy_Returns'].resample('M').sum()
        monthly_returns_pivot = monthly_returns.to_frame()
        monthly_returns_pivot['Year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['Month'] = monthly_returns_pivot.index.month
        heatmap_data = monthly_returns_pivot.pivot(index='Month', columns='Year', values='Strategy_Returns')
        sns.heatmap(heatmap_data * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, ax=ax6, cbar_kws={'label': 'Return (%)'})
        ax6.set_title('Monthly Returns Heatmap (%)')
        ax6.set_ylabel('Month')
        
        plt.tight_layout()
        plt.show()
        
    def run_full_pipeline(self, optimize=False):
        """Run the complete pipeline"""
        print("\n" + "="*50)
        print("STARTING FULL QUANT PIPELINE")
        print("="*50)
        
        # 1. Fetch data
        self.fetch_data()
        
        # 2. Create features
        self.create_technical_features()
        
        # 3. Create target
        self.create_target(forward_days=1, threshold=0.0)
        
        # 4. Prepare train/test split
        self.prepare_train_test(train_split=0.8)
        
        # 5. Train model
        self.train_model(optimize=optimize)
        
        # 6. Evaluate model
        self.evaluate_model()
        
        # 7. Backtest
        self.backtest(initial_capital=100000, transaction_cost=0.001)
        
        # 8. Plot results
        self.plot_results()
        
        print("\n" + "="*50)
        print("PIPELINE COMPLETE!")
        print("="*50)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("XGBOOST QUANTITATIVE TRADING ALGORITHM")
    print("Fetching real market data from Yahoo Finance...")
    print("="*70)
    
    # Initialize and run the strategy with real market data
    strategy = QuantXGBoostStrategy(
        ticker='SPY',  # S&P 500 ETF
        start_date='2018-01-01',  # 6+ years of data
        end_date=datetime.now().strftime('%Y-%m-%d')  # Today's date
    )
    
    # Run full pipeline (set optimize=True for hyperparameter tuning, but it takes longer)
    strategy.run_full_pipeline(optimize=False)
    
    # Access specific results
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES FOR PREDICTIONS:")
    print("="*70)
    print(strategy.feature_importance.head(10).to_string(index=False))
    
    # Show sample predictions
    print("\n" + "="*70)
    print("RECENT PREDICTIONS (Last 10 Trading Days):")
    print("="*70)
    recent = strategy.backtest_df.tail(10)[['Date', 'Price', 'Prediction', 'Probability', 'Signal']]
    recent['Prediction'] = recent['Prediction'].map({1: 'UP', 0: 'DOWN'})
    recent['Signal'] = recent['Signal'].map({1: 'BUY', 0: 'HOLD'})
    print(recent.to_string(index=False))
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - Using Real Market Data from Yahoo Finance")
    print("="*70)
