""" Reward functions for planning.

These reward functions should take a Trajectory object and either an Environment
or Model object as input and return a single scalar value. The Trajectory object
should support the function `uniformly_sample`, called with arguments contained
in params.

NOTE: I made this class based so we can add other properties
as needed (e.g., reward gradients)
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import rankdata          # empirical CDF helper



class Reward(ABC):
    """Abstract Reward base class."""

    def __init__(self, **kwargs):
        """Initialize reward object."""
        pass

    @abstractmethod
    def eval(self, trajectory, env_model, **kwargs):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`

        Returns: (float) reward value
        """
        pass


class SampleValues(Reward):
    """Counts the total value of the samples."""

    def __init__(self, sampling_params={}, is_cost=False):
        """Initialize reward object.

        Args:
            sampling_params (dict): a dictionary of named paramters with which to call
                `uniformly_sample`. Defaults to an empty ditionary.
            is_cost (bool): if True, returns a cost instead of reward value
        """
        self.params = sampling_params
        self.is_cost = is_cost

    def _json_stats(self):
        """Returns dict of reward info."""
        json_dict = {"reward_func": "SampleValues",
                     "sampling_params": self.params,
                     "is_cost": self.is_cost}
        return json_dict

    def eval(self, trajectory, env_model, from_cache=False):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`

        Returns: (float) reward value
        """
        # Get sample points
        samples = np.asarray(trajectory.uniformly_sample(**self.params))
        #print(f"samples shape: {samples.shape}")

        # Grab the reward from the at a specific snapshot time
        # Assumes that a snapshot at the start of the trajectory
        # are the same.

        vals = env_model.get_value(t=trajectory.t0, loc=(
            samples[:, 1], samples[:, 2], samples[:, 3]), from_cache=from_cache)
        reward = 1e4 * float(vals.sum())

        if self.is_cost:
            return -1.0 * reward
        return reward


class SampleValuesPrioritizeMid(Reward):
    """Rewards sampling points that prioritize values near the optimal threshold between
    high and low confidence regions along the trajectory."""

    def __init__(self, sampling_params=None, scale=1e4, is_cost=False, threshold=None, threshold_shift=None):
        """Initialize reward object.

        Args:
            sampling_params (dict): Parameters for uniformly_sample. Defaults to empty dict.
            scale (float): Scaling factor for the reward. Defaults to 1e4.
            is_cost (bool): If True, returns a cost instead of reward value.
        """
        self.params = sampling_params or {}
        self.scale = scale
        self.is_cost = is_cost
        self.threshold = threshold
        self.threshold_shift = threshold_shift
    def _json_stats(self):
        """Returns dict of reward info."""
        return {"reward_func": "SampleValuesPrioritizeMid",
                "sampling_params": self.params,
                "scale": self.scale,
                "is_cost": self.is_cost,
                "threshold": self.threshold}
    
    def _otsu_threshold(self, values):
        """Compute optimal threshold using Otsu's method.
        
        Args:
            values (numpy.ndarray): Array of confidence values
            
        Returns:
            float: Optimal threshold value
        """
        # Create histogram of values
        min_val, max_val = np.min(values), np.max(values)
        if min_val == max_val:  # Handle edge case
            return min_val
            
        # Use 64 bins for the histogram
        hist, bin_edges = np.histogram(values, bins=64, range=(min_val, max_val))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Total number of pixels
        total = np.sum(hist)
        if total == 0:  # Handle edge case
            return min_val
            
        # Calculate sum and squared sum
        sum_total = np.sum(hist * bin_centers)
        
        # Initialize variables
        best_threshold = 0
        best_variance = 0
        
        # Values for each iteration
        weight_bg = 0
        sum_bg = 0
        
        # Check each possible threshold
        for i, threshold in enumerate(bin_centers):
            # Update background class statistics
            weight_bg += hist[i]
            sum_bg += hist[i] * bin_centers[i]
            
            # Skip if background or foreground class is empty
            if weight_bg == 0 or weight_bg == total:
                continue
                
            # Calculate foreground class statistics
            weight_fg = total - weight_bg
            sum_fg = sum_total - sum_bg
            
            # Calculate means
            mean_bg = sum_bg / weight_bg
            mean_fg = sum_fg / weight_fg
            
            # Calculate between-class variance
            variance = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
            
            # Update best threshold if we found better variance
            if variance > best_variance:
                best_variance = variance
                best_threshold = threshold
                
        return best_threshold

    def eval(self, trajectory, env_model, from_cache=False):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environment object,
                supporting method `get_val`

        Returns: (float) reward value
        """
            
        # Get sample points
        samples = np.asarray(trajectory.uniformly_sample(**self.params))
        vals = env_model.get_value(t=trajectory.t0, loc=(
            samples[:, 1], samples[:, 2], samples[:, 3]), from_cache=from_cache)

        # Calculate how close each sample is to the threshold
        proximity_to_threshold = (np.abs(vals - self.threshold) - self.threshold_shift) 
        
        # Average reward across all sample points
        avg_reward = -1 * float(proximity_to_threshold.sum())
        
        # Apply scaling
        final_reward = self.scale * avg_reward
        
        if self.is_cost:
            return -1.0 * final_reward
        return final_reward


class SampleUCB(Reward):
    """Counts the total UCB value of the samples."""

    def __init__(self, sampling_params={}, is_cost=False, c=1.0):
        """Initialize reward object.

        Args:
            sampling_params (dict): a dictionary of named paramters with which to call
                `uniformly_sample`. Defaults to an empty ditionary.
            is_cost (bool): if True, returns a cost instead of reward value
        """
        self.params = sampling_params
        self.is_cost = is_cost
        self.c = c

    def _json_stats(self):
        """Returns dict of reward info."""
        json_dict = {"reward_func": "SampleUCB",
                     "sampling_params": self.params,
                     "is_cost": self.is_cost,
                     "c": self.c}
        return json_dict

    def eval(self, trajectory, env_model, from_cache=False):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`

        Returns: (float) reward value
        """

        # Get sample points
        samples = np.asarray(trajectory.uniformly_sample(**self.params))

        # Grab the reward from the at a specific snapshot time
        # Assumes that a snapshot at the start of the trajectory
        # are the same.

        mean, var = env_model.get_prediction(t=trajectory.t0, loc=(
            samples[:, 1], samples[:, 2], samples[:, 3]), from_cache=from_cache)

        # UCB reward
        reward = 1e4 * (float(mean.sum()) + self.c * float(var.sum()))
        if self.is_cost:
            return -1.0 * reward
        return reward

# Edits
class SampleUCBEdited(Reward):
    """Counts the total UCB value of the samples."""

    def __init__(self, sampling_params={}, is_cost=False, c=1.0):
        """Initialize reward object.

        Args:
            sampling_params (dict): a dictionary of named paramters with which to call
                `uniformly_sample`. Defaults to an empty ditionary.
            is_cost (bool): if True, returns a cost instead of reward value
        """
        self.params = sampling_params
        self.is_cost = is_cost
        self.c = c

    def _json_stats(self):
        """Returns dict of reward info."""
        json_dict = {"reward_func": "SampleUCB",
                     "sampling_params": self.params,
                     "is_cost": self.is_cost,
                     "c": self.c}
        return json_dict

    def eval(self, trajectory, env_model, from_cache=False):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`

        Returns: (float) reward value
        """

        # Get sample points
        samples = np.asarray(trajectory.uniformly_sample(**self.params))

        # Grab the reward from the at a specific snapshot time
        # Assumes that a snapshot at the start of the trajectory
        # are the same.

        mean, var = env_model.get_prediction(t=trajectory.t0, loc=(
            samples[:, 1], samples[:, 2], samples[:, 3]), from_cache=from_cache)
        print(f"reward mean: {mean}")

        # Rewards values near 0.5 and high variance
        reward = 1e4 * (1 - (float(mean.sum())-0.5)**2 + self.c * float(var.sum()))
        if self.is_cost:
            return -1.0 * reward
        return reward
