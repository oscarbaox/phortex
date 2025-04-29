"""Base classes for metrics."""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class OptimizationMetric(ABC):
    """Abstract base class for optimization metrics."""
    
    def __init__(self, **kwargs):
        """Initialize the metric."""
        pass

    @abstractmethod
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        """
        Evaluate the quality of optimization using this metric
        
        Args:
            trajectory: Final trajectory produced by optimizer
            env_model: The model used for planning
            true_environment: Ground truth environment
            reward_history: History of rewards during optimization
            **kwargs: Additional metric-specific parameters
            
        Returns:
            dict: Metric results with relevant statistics
        """
        pass
    
    def __str__(self):
        """String representation of the metric."""
        return self.__class__.__name__

class MetricVisualizer:
    """Utility class for visualizing metric results."""
    
    @staticmethod
    def plot_metric_history(metric_results, save_path):
        """Plot metric results over time/iterations.
        
        Args:
            metric_results (dict): Dictionary of metric results
            save_path (str): Path to save the visualization
        """
        if not metric_results:
            print("No metric results to visualize")
            return
            
        fig, axes = plt.subplots(len(metric_results), 1, figsize=(10, 4*len(metric_results)))
        
        # Handle case of single metric
        if len(metric_results) == 1:
            axes = [axes]
            
        for ax, (metric_name, results) in zip(axes, metric_results.items()):
            for stat_name, value in results.items():
                try:
                    if isinstance(value, (list, np.ndarray)):
                        ax.plot(value, label=stat_name)
                    else:
                        # For scalar values, add a horizontal line with value in legend
                        ax.axhline(y=float(value), linestyle='--', 
                                  label=f"{stat_name}: {float(value):.4f}")
                except (ValueError, TypeError) as e:
                    print(f"Could not plot {stat_name}: {e}")
                    continue
            ax.set_title(f'{metric_name}')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 