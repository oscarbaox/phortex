"""Standard metric implementations."""

import numpy as np
from .base import OptimizationMetric

class CoverageMetric(OptimizationMetric):
    """Metric that evaluates trajectory coverage and prediction accuracy."""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        """
        Evaluate trajectory coverage and model prediction accuracy.
        1. Creates a grid over the trajectory's bounding box
        2. For each point (x,y,z) in the grid:
           - Gets true plume value from ground truth
           - Gets predicted value from model
        3. Calculates:
           - True positive rate = % of points above threshold
           - Prediction accuracy = mean absolute error
           - Coverage area = total points sampled
        
        Args:
            trajectory: Trajectory to evaluate
            env_model: Environment model used for planning
            true_environment: Ground truth environment
            reward_history: History of rewards during optimization
            **kwargs: Additional parameters including sampling_params and threshold
            
        Returns:
            dict: Metrics including true positive rate, prediction accuracy, and coverage
        """
        # Get sampling parameters
        samp_dist = kwargs.get('samp_dist', 1.0)  # Default to 1.0 if not provided
        threshold = kwargs.get('threshold', 1e-5)
        
        try:
            # Sample extent of environment
            xmin, xmax = trajectory.xmin, trajectory.xmax
            ymin, ymax = trajectory.ymin, trajectory.ymax
            z = trajectory.altitude
            
            # Create grid for evaluation
            x_grid = np.linspace(xmin, xmax, int((xmax - xmin) / samp_dist) + 1)
            y_grid = np.linspace(ymin, ymax, int((ymax - ymin) / samp_dist) + 1)
            xx, yy = np.meshgrid(x_grid, y_grid)
            zz = np.ones_like(xx) * z
            
            # Flatten for sampling
            xs = xx.flatten()
            ys = yy.flatten()
            zs = zz.flatten()
            
            # Get values from true environment and model
            true_vals = true_environment.get_value(t=trajectory.t0, loc=(xs, ys, zs))
            predicted_vals = env_model.get_value(t=trajectory.t0, loc=(xs, ys, zs))
            
            # Calculate metrics
            return {
                'true_positive_rate': float(np.mean(true_vals > threshold)),
                'prediction_accuracy': float(np.mean(np.abs(true_vals - predicted_vals))),
                'coverage_area': int(len(xs)),
                'trajectory_length': float(trajectory.length)
            }
        except Exception as e:
            print(f"Error in CoverageMetric: {str(e)}")
            # Return placeholder values
            return {
                'true_positive_rate': 0.0,
                'prediction_accuracy': 0.0,
                'coverage_area': 0,
                'trajectory_length':0.0
            }


class ConvergenceMetric(OptimizationMetric):
    """Metric that evaluates optimization convergence performance."""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        """
        Evaluate optimization convergence performance.
        1. Takes reward history from optimization
        2. Calculates:
           - Convergence rate = std(diff(rewards)) - measures optimization stability
           - Final reward = last reward value
           - Improvement ratio = final_reward / initial_reward
        
        Args:
            trajectory: Final trajectory produced by optimizer
            env_model: Model used for planning
            true_environment: Ground truth environment
            reward_history: History of rewards during optimization
            **kwargs: Additional parameters
            
        Returns:
            dict: Convergence metrics including rate, final reward, and improvement ratio
        """
        if not reward_history or len(reward_history) < 2:
            return {
                'convergence_rate': 0.0,
                'final_reward': reward_history[-1] if reward_history else 0.0,
                'improvement_ratio': 1.0,
                'best_reward': 0.0,
                'drift_ratio': 1.0
            }
            
        # Calculate convergence statistics
        return {
            'convergence_rate': float(np.std(np.diff(reward_history))),
            'final_reward': float(reward_history[-1]),
            'improvement_ratio': float(reward_history[-1] / reward_history[0]) if reward_history[0] != 0 else 1.0,
            'best_reward': float(min_cost[1]),
            'drift_ratio': float(reward_history[-1] / min_cost[1]) if min_cost[1] != 0 else 1.0
        }


class ComputationalEfficiencyMetric(OptimizationMetric):
    """Metric that evaluates computational efficiency of optimization."""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        """
        Evaluate computational efficiency of optimization.
         1. Takes reward history
         2. Calculates:
            - Iterations to converge = length of history
            - Average improvement = mean(diff(rewards))
        
        Args:
            trajectory: Final trajectory produced by optimizer
            env_model: Model used for planning
            true_environment: Ground truth environment
            reward_history: History of rewards during optimization
            **kwargs: Additional parameters
            
        Returns:
            dict: Efficiency metrics including iterations and average improvement
        """
        if not reward_history or len(reward_history) < 2:
            return {
                'iterations_to_converge': len(reward_history) if reward_history else 0,
                'average_improvement_per_step': 0.0
            }
            
        # Calculate efficiency statistics
        return {
            'iterations_to_converge': len(reward_history),
            'average_improvement_per_step': float(np.mean(np.diff(reward_history)))
        } 
    
class RobustnessMetric(OptimizationMetric):
    """Evaluates trajectory robustness to perturbations"""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        # Add small perturbations to trajectory parameters
        perturbation = 0.05  # 5% perturbation
        original_value = env_model.get_value(t=trajectory.t0, 
                                        loc=(trajectory.xcoords, 
                                                trajectory.ycoords, 
                                                trajectory.altitude))
        
        perturbed_values = []
        for _ in range(10):  # Test 10 perturbations
            perturbed_coords = (trajectory.xcoords * (1 + np.random.normal(0, perturbation)),
                            trajectory.ycoords * (1 + np.random.normal(0, perturbation)),
                            trajectory.altitude)
            
            perturbed_value = env_model.get_value(t=trajectory.t0, loc=perturbed_coords)
            perturbed_values.append(perturbed_value)
            
        return {
            'stability_score': float(np.std(perturbed_values) / np.mean(perturbed_values)),
            'worst_case_deviation': float(np.max(np.abs(perturbed_values - original_value))),
            'mean_sensitivity': float(np.mean(np.abs(perturbed_values - original_value)))
        }
class InformationGainMetric(OptimizationMetric):
    """Evaluates the information gained during trajectory execution"""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        # Sample points along trajectory
        points = trajectory.uniformly_sample(kwargs.get('samp_dist', 1.0))
        points = np.array(points)
        
        # Get model uncertainties at start and end
        initial_uncertainty = env_model.get_prediction(t=trajectory.t0, 
                                                    loc=(points[:, 1], points[:, 2], points[:, 3]))[1]
        final_uncertainty = env_model.get_prediction(t=trajectory.time_at_end,
                                                loc=(points[:, 1], points[:, 2], points[:, 3]))[1]
        
        return {
            'uncertainty_reduction': float(np.mean(initial_uncertainty - final_uncertainty)),
            'final_entropy': float(np.mean(final_uncertainty)),
            'information_gain_rate': float(np.mean(initial_uncertainty - final_uncertainty) / trajectory.length)
        }

class SpatialEfficiencyMetric(OptimizationMetric):
    """Evaluates the spatial efficiency of the trajectory"""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        # Get trajectory points
        points = trajectory.uniformly_sample(kwargs.get('samp_dist', 1.0))
        points = np.array(points)
        
        # Calculate metrics
        path_length = trajectory.length
        area_covered = (trajectory.xmax - trajectory.xmin) * (trajectory.ymax - trajectory.ymin)
        
        # Calculate trajectory curvature
        dx = np.diff(points[:, 1])  # x coordinates
        dy = np.diff(points[:, 2])  # y coordinates
        angles = np.arctan2(dy, dx)
        curvature = np.abs(np.diff(angles))
        
        return {
            'coverage_efficiency': float(area_covered / path_length),
            'path_smoothness': float(np.mean(curvature)),
            'space_utilization': float(len(points) / area_covered)
        }

class PlumeDynamicsMetric(OptimizationMetric):
    """Evaluates how well the trajectory adapts to plume dynamics"""
    
    def evaluate(self, trajectory, env_model, true_environment, reward_history, min_cost, **kwargs):
        # Sample at different time steps
        times = np.linspace(trajectory.t0, trajectory.time_at_end, 10)
        dynamics_scores = []
        
        for t in times:
            # Get plume center at this time
            true_center = true_environment.get_maxima(t)
            pred_center = env_model.get_maxima(t)
            
            # Calculate center tracking error
            center_error = np.linalg.norm(np.array(true_center) - np.array(pred_center))
            dynamics_scores.append(center_error)
            
        return {
            'center_tracking_error': float(np.mean(dynamics_scores)),
            'temporal_variance': float(np.std(dynamics_scores)),
            'max_deviation': float(np.max(dynamics_scores))
        }