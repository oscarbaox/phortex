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
    """Rewards sampling points that prioritize values near the median of the distribution."""

    def __init__(self, sampling_params=None, scale=1e4, is_cost=False):
        """Initialize reward object.

        Args:
            sampling_params (dict): Parameters for uniformly_sample. Defaults to empty dict.
            scale (float): Scaling factor for the reward. Defaults to 1e4.
            is_cost (bool): If True, returns a cost instead of reward value.
        """
        self.params = sampling_params or {}
        self.scale = scale
        self.is_cost = is_cost

    def _json_stats(self):
        """Returns dict of reward info."""
        return {"reward_func": "SampleValuesPrioritizeMid",
                "sampling_params": self.params,
                "scale": self.scale,
                "is_cost": self.is_cost}

    def eval(self, trajectory, env_model, from_cache=False):
        """Evaluate the reward of a trajectory and model or environment.

        Args:
            trajectory (Trajectory): a Trajectory object
            env_model (Model/Environment): a Model or Environment object,
                supporting method `get_val`

        Returns: (float) reward value
        """
        # Get sample points along trajectory
        samples = np.asarray(trajectory.uniformly_sample(**self.params))
        
        # Get values at sample points
        vals = env_model.get_value(
            t=trajectory.t0,
            loc=(samples[:, 1], samples[:, 2], samples[:, 3]),
            from_cache=from_cache
        ).astype(float)
        
        # Calculate percentiles from the values directly
        # We can use the values themselves to establish the empirical CDF
        # This is more efficient than creating a separate grid
        sorted_vals = np.sort(vals)
        percentiles = rankdata(vals, method="average", axis=None) / len(vals)
        
        # Calculate mean squared error from target percentile (0.5)
        # Lower MSE means values are closer to median - this is what we want to reward
        mse = float(np.mean((percentiles - 0.5) ** 2))
        
        # Convert MSE to reward (smaller MSE = higher reward)
        reward = self.scale * (1.0 - mse)
        
        if self.is_cost:
            return -1.0 * reward
        return reward


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
