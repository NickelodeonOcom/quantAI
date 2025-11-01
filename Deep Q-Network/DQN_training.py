import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import logging


@dataclass
class TrainingConfig:
    """Configuration for DQN training."""
    # Model hyperparameters
    learning_rate: float = 5e-5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 500
    hidden_dims: List[int] = None
    
    # Training parameters
    n_episodes: int = 100
    max_steps_per_episode: int = 10000
    warmup_episodes: int = 10
    eval_frequency: int = 5
    save_frequency: int = 10
    
    # Environment parameters
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    lookback_window: int = 30
    max_position_per_asset: float = 0.3
    
    # Optimization
    use_prioritized_replay: bool = False
    use_double_dqn: bool = True
    use_dueling_dqn: bool = False
    gradient_clip: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_threshold: float = 0.0
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
    
    def save(self, filepath: str):
        """Save configuration to JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=4)
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class TrainingLogger:
    """Logger for training metrics and events."""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.episode_metrics = []
        self.training_metrics = []
    
    def log_episode(self, episode: int, metrics: Dict):
        """Log episode metrics."""
        metrics['episode'] = episode
        metrics['timestamp'] = datetime.now().isoformat()
        self.episode_metrics.append(metrics)
        
        self.logger.info(
            f"Episode {episode}: "
            f"Reward={metrics.get('reward', 0):.4f}, "
            f"Value=${metrics.get('final_value', 0):,.2f}, "
            f"Sharpe={metrics.get('sharpe', 0):.3f}, "
            f"Epsilon={metrics.get('epsilon', 0):.4f}"
        )
    
    def log_training_step(self, step: int, loss: float):
        """Log training step."""
        self.training_metrics.append({
            'step': step,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        })
    
    def save_metrics(self):
        """Save all metrics to CSV files."""
        if self.episode_metrics:
            df = pd.DataFrame(self.episode_metrics)
            filepath = os.path.join(self.log_dir, 'episode_metrics.csv')
            df.to_csv(filepath, index=False)
            self.logger.info(f"Episode metrics saved to {filepath}")
        
        if self.training_metrics:
            df = pd.DataFrame(self.training_metrics)
            filepath = os.path.join(self.log_dir, 'training_metrics.csv')
            df.to_csv(filepath, index=False)
            self.logger.info(f"Training metrics saved to {filepath}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, threshold: float = 0.0):
        self.patience = patience
        self.threshold = threshold
        self.best_score = -np.inf
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current performance score (higher is better)
        
        Returns:
            True if training should stop
        """
        if score > self.best_score + self.threshold:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, save_dir: str = 'checkpoints', keep_last_n: int = 5):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.checkpoints = []
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, agent, episode: int, metrics: Dict):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_ep{episode}_{timestamp}.pth'
        filepath = os.path.join(self.save_dir, filename)
        
        # Save model and metrics
        torch.save({
            'episode': episode,
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'steps': agent.steps,
            'metrics': metrics
        }, filepath)
        
        self.checkpoints.append(filepath)
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        return filepath
    
    def save_best_model(self, agent, metrics: Dict):
        """Save best performing model."""
        filepath = os.path.join(self.save_dir, 'best_model.pth')
        torch.save({
            'policy_net_state_dict': agent.policy_net.state_dict(),
            'target_net_state_dict': agent.target_net.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'steps': agent.steps,
            'metrics': metrics
        }, filepath)
        return filepath


class TrainingPipeline:
    """Complete training pipeline for DQN agent."""
    
    def __init__(
        self,
        env,
        agent,
        config: TrainingConfig,
        logger: Optional[TrainingLogger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logger or TrainingLogger()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            threshold=config.early_stopping_threshold
        )
        
        self.best_sharpe = -np.inf
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'losses': []
        }
    
    def train_episode(self, episode: int) -> Dict:
        """Train for one episode."""
        state = self.env.reset()
        episode_reward = 0
        episode_losses = []
        steps = 0
        
        while True:
            # Select action
            training = episode >= self.config.warmup_episodes
            action = self.agent.select_action(state, training=training)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if training:
                loss = self.agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    self.logger.log_training_step(self.agent.steps, loss)
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done or steps >= self.config.max_steps_per_episode:
                break
        
        # Calculate episode metrics
        metrics = {
            'reward': episode_reward,
            'final_value': info['portfolio_value'],
            'sharpe': info['sharpe_ratio'],
            'trades': info['total_trades'],
            'avg_loss': np.mean(episode_losses) if episode_losses else 0,
            'epsilon': self.agent.epsilon,
            'steps': steps
        }
        
        return metrics
    
    def evaluate(self, n_eval_episodes: int = 3) -> Dict:
        """Evaluate agent performance."""
        eval_rewards = []
        eval_sharpes = []
        eval_values = []
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # No exploration during evaluation
        
        for _ in range(n_eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_sharpes.append(info['sharpe_ratio'])
            eval_values.append(info['portfolio_value'])
        
        self.agent.epsilon = original_epsilon
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_sharpe': np.mean(eval_sharpes),
            'avg_value': np.mean(eval_values),
            'std_reward': np.std(eval_rewards),
            'std_sharpe': np.std(eval_sharpes)
        }
    
    def train(self):
        """Run complete training pipeline."""
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("Starting DQN Training Pipeline")
        self.logger.logger.info("=" * 80)
        self.logger.logger.info(f"Device: {self.config.device}")
        self.logger.logger.info(f"Episodes: {self.config.n_episodes}")
        self.logger.logger.info(f"Assets: {', '.join(self.env.symbols)}")
        
        # Save configuration
        config_path = os.path.join(self.logger.log_dir, 'config.json')
        self.config.save(config_path)
        self.logger.logger.info(f"Configuration saved to {config_path}")
        
        try:
            for episode in range(1, self.config.n_episodes + 1):
                # Train episode
                metrics = self.train_episode(episode)
                
                # Log metrics
                self.logger.log_episode(episode, metrics)
                
                # Store history
                self.training_history['episodes'].append(episode)
                self.training_history['rewards'].append(metrics['reward'])
                self.training_history['portfolio_values'].append(metrics['final_value'])
                self.training_history['sharpe_ratios'].append(metrics['sharpe'])
                self.training_history['losses'].append(metrics['avg_loss'])
                
                # Periodic evaluation
                if episode % self.config.eval_frequency == 0:
                    eval_metrics = self.evaluate()
                    self.logger.logger.info(
                        f"Evaluation: "
                        f"Avg Reward={eval_metrics['avg_reward']:.4f}, "
                        f"Avg Sharpe={eval_metrics['avg_sharpe']:.3f}"
                    )
                    
                    # Check for best model
                    if eval_metrics['avg_sharpe'] > self.best_sharpe:
                        self.best_sharpe = eval_metrics['avg_sharpe']
                        filepath = self.checkpoint_manager.save_best_model(
                            self.agent, 
                            {**metrics, **eval_metrics}
                        )
                        self.logger.logger.info(
                            f"New best model saved! Sharpe: {self.best_sharpe:.3f}"
                        )
                    
                    # Early stopping check
                    if self.early_stopping(eval_metrics['avg_sharpe']):
                        self.logger.logger.info(
                            f"Early stopping triggered at episode {episode}"
                        )
                        break
                
                # Save checkpoint
                if episode % self.config.save_frequency == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.agent, 
                        episode, 
                        metrics
                    )
        
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        
        finally:
            # Save final metrics
            self.logger.save_metrics()
            self.logger.logger.info("Training completed!")
            self.logger.logger.info(f"Best Sharpe Ratio: {self.best_sharpe:.3f}")
        
        return self.training_history
    
    def plot_training_progress(self):
        """Plot training progress."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.training_history['episodes'], 
                       self.training_history['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Portfolio Values
        axes[0, 1].plot(self.training_history['episodes'], 
                       self.training_history['portfolio_values'])
        axes[0, 1].axhline(y=self.config.initial_balance, 
                          color='r', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Portfolio Value')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Value ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sharpe Ratios
        axes[1, 0].plot(self.training_history['episodes'], 
                       self.training_history['sharpe_ratios'])
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Sharpe Ratio')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training Loss
        axes[1, 1].plot(self.training_history['episodes'], 
                       self.training_history['losses'])
        axes[1, 1].set_title('Average Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.logger.log_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.logger.info(f"Training progress plot saved to {save_path}")
        
        return fig


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Production DQN Training Pipeline")
    print("=" * 80)
    print("\nFeatures:")
    print("  ✓ Comprehensive configuration management")
    print("  ✓ Structured logging and metrics tracking")
    print("  ✓ Checkpoint management with best model saving")
    print("  ✓ Early stopping mechanism")
    print("  ✓ Periodic evaluation during training")
    print("  ✓ Training progress visualization")
    print("  ✓ Exception handling and graceful shutdown")
    print("\nUsage:")
    print("  config = TrainingConfig(n_episodes=100, learning_rate=5e-5)")
    print("  pipeline = TrainingPipeline(env, agent, config)")
    print("  history = pipeline.train()")
    print("  pipeline.plot_training_progress()")
