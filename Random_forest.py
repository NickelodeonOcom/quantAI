import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Node:
    """Decision tree node"""
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    """Decision Tree for regression"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, max_features: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Build decision tree"""
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively grow the tree"""
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            return Node(value=np.mean(y))
        
        # Find best split
        feat_idxs = np.random.choice(n_features, self.max_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        if best_feat is None:
            return Node(value=np.mean(y))
        
        # Split and recurse
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        
        # Check minimum samples per leaf
        if np.sum(left_idxs) < self.min_samples_leaf or np.sum(right_idxs) < self.min_samples_leaf:
            return Node(value=np.mean(y))
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, 
                    feat_idxs: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best feature and threshold to split on"""
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = threshold
        
        return best_feat, best_thresh
    
    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, 
                         threshold: float) -> float:
        """Calculate variance reduction (information gain for regression)"""
        parent_var = np.var(y)
        
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = np.sum(left_idxs), np.sum(right_idxs)
        var_l, var_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        
        child_var = (n_l / n) * var_l + (n_r / n) * var_r
        
        return parent_var - child_var
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for samples"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """Traverse tree to get prediction"""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForest:
    """Random Forest for quantitative trading"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, bootstrap: bool = True,
                 random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees: List[DecisionTree] = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """Train the random forest"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples = X.shape[0]
        
        # Determine max_features
        if self.max_features is None:
            max_features = max(1, int(np.sqrt(X.shape[1])))
        else:
            max_features = self.max_features
        
        self.trees = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[idxs], y[idxs]
            else:
                X_sample, y_sample = X, y
            
            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by averaging tree predictions"""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds, axis=0)
    
    def feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Calculate feature importance (simplified version)"""
        # This is a placeholder - full implementation would track splits
        n_features = self.trees[0].n_features
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        importances = np.random.rand(n_features)  # Placeholder
        importances /= importances.sum()
        
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


class QuantRandomForest:
    """Random Forest specifically designed for quantitative trading"""
    
    def __init__(self, lookback_period: int = 20, n_estimators: int = 100,
                 max_depth: int = 8, random_state: Optional[int] = 42):
        self.lookback_period = lookback_period
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trading features from price data"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
        
        # Volatility
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Momentum indicators
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_sma_20'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        
        # High-low range
        if 'high' in df.columns and 'low' in df.columns:
            features['hl_range'] = (df['high'] - df['low']) / df['close']
            features['hl_range_sma_10'] = features['hl_range'].rolling(10).mean()
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = features['returns'].shift(lag)
        
        self.feature_names = features.columns.tolist()
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self, df: pd.DataFrame, 
                    forward_returns: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variable"""
        features_df = self.create_features(df)
        
        # Target: forward returns
        target = df['close'].pct_change(forward_returns).shift(-forward_returns)
        
        # Combine and drop NaN
        data = pd.concat([features_df, target.rename('target')], axis=1)
        data = data.dropna()
        
        X = data[self.feature_names].values
        y = data['target'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, forward_returns: int = 1,
             train_size: float = 0.8) -> dict:
        """Train the model and return performance metrics"""
        X, y = self.prepare_data(df, forward_returns)
        
        # Train-test split
        split_idx = int(len(X) * train_size)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model = RandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        
        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Metrics
        train_mse = np.mean((y_train - train_pred) ** 2)
        test_mse = np.mean((y_test - test_pred) ** 2)
        train_r2 = 1 - (np.sum((y_train - train_pred) ** 2) / 
                       np.sum((y_train - np.mean(y_train)) ** 2))
        test_r2 = 1 - (np.sum((y_test - test_pred) ** 2) / 
                      np.sum((y_test - np.mean(y_test)) ** 2))
        
        return {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features_df = self.create_features(df)
        X = features_df[self.feature_names].values
        
        # Handle NaN values by forward filling
        X = pd.DataFrame(X, columns=self.feature_names).fillna(method='ffill').values
        
        return self.model.predict(X)
    
    def generate_signals(self, df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """Generate trading signals based on predictions"""
        predictions = self.predict(df)
        
        signals = pd.DataFrame(index=df.index)
        signals['prediction'] = np.nan
        signals.iloc[-len(predictions):, signals.columns.get_loc('prediction')] = predictions
        
        signals['signal'] = 0
        signals.loc[signals['prediction'] > threshold, 'signal'] = 1  # Buy
        signals.loc[signals['prediction'] < -threshold, 'signal'] = -1  # Sell
        
        return signals


# Example usage
if __name__ == "__main__":
    # Generate synthetic price data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    price = 100
    prices = [price]
    for _ in range(999):
        price *= (1 + np.random.randn() * 0.02)
        prices.append(price)
    
    df = pd.DataFrame({
        'close': prices,
        'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
        'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
        'volume': np.random.randint(1000000, 5000000, 1000)
    }, index=dates)
    
    # Initialize and train
    qrf = QuantRandomForest(n_estimators=50, max_depth=6, random_state=42)
    
    print("Training Random Forest model...")
    metrics = qrf.train(df, forward_returns=5, train_size=0.8)
    
    print("\n=== Model Performance ===")
    print(f"Train MSE: {metrics['train_mse']:.6f}")
    print(f"Test MSE: {metrics['test_mse']:.6f}")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Train Samples: {metrics['train_samples']}")
    print(f"Test Samples: {metrics['test_samples']}")
    
    # Generate signals
    signals = qrf.generate_signals(df, threshold=0.001)
    
    print("\n=== Trading Signals (Last 10) ===")
    print(signals.tail(10))
    
    print("\n=== Signal Distribution ===")
    print(signals['signal'].value_counts())
