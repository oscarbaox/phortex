import numpy as np
from scipy import optimize
from scipy.optimize import minimize

import os
import json
import matplotlib.pyplot as plt

from fumes.utils import tic, toc
from fumes.simulator.utils import visualize_and_save_traj
from fumes.metrics import MetricVisualizer

from .planner import Planner
from .utils import length_constraint, bound_constraint, param_constraint, \
    soft_origin_penalty, reconstruct_theta


class TrajectoryOpt(Planner):
    def __init__(self, env_model, traj_generator, reward, x0, budget=None,
                 limits=[0., 1000., 0., 1000.], param_bounds=None,
                 param_names=None, max_iters=30, tol=1e-8,
                 method="trust-constr", experiment_name=None, hierarchy=False,
                 initial_params=[None,None,None,None,None],metrics=None,
                 scaling=[1,1,1,1,1],adj_secondary=[1,1,1,1,1]):
        """ Initialize trajectory optmizer.

        Args:
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`
            traj_generator (TrajectoryGenerator): a TrajectoryGenerator object
            reward (Reward): a Reward object that takes a Trajectory as input
                and outputs a reward score
            x0 (np.array): input parameter values that can be fed to the
                traj_generator object to produce a trajectory, e.g.,
                np.array([height, width, theta, resolution])
            budget (float): if a length budget is provided, the returned
                trajectory will be constrained
            limits (tuple[float]): axis-aligned safety zone, in x and y,
                e.g., (xmin, xmax, ymin, ymax)
            param_bounds (list[tuple]): list of upper and lower bounds
                for parameters
            param_names (dict): dictionary from parameter names to index
                in parameter vector
            max_iters (int): maximum number of optimization iterations
            tol (float): algorithm convergence tolerance
            method (str): optimization method, one of:
                "SLSQP", "trust-constr", "BFGS", "basinhopping"
            experiment_name (str): the name of the experiment (optional), used
                to name output files.
            metrics (list[OptimizationMetric]): list of metrics to evaluate
                during optimization
        """
        self.env_model = env_model
        self.traj_generator = traj_generator
        self.reward = reward
        self.x0 = x0
        self.budget = budget
        self.limits = limits
        self.method = method
        self.param_bounds = param_bounds
        self.param_names = param_names
        self.max_iters = max_iters
        self.tol = tol
        self.experiment_name = experiment_name
        self.reward_history = []
        self.hierarchy = hierarchy
        self.initial_params = initial_params
        self.constant_params = initial_params
        self.metrics = metrics or []  # Initialize empty list if no metrics provided
        self.scaling = scaling
        self.adj_secondary = adj_secondary
        self.min_cost = [None,None]


        if self.experiment_name is None:
            self.experiment_name = "temp"
        self.id = ""

        self.neval = 1

        # Generate planning path, if needed
        self.path = os.path.join(os.getenv("FUMES_OUTPUT"), "planning", self.experiment_name)
        os.makedirs(self.path, exist_ok=True)

    def _json_stats(self):
        """Returns a dict of info about this optimizer."""
        json_dict = {"x0": self.x0,
                     "budget": self.budget,
                     "limits": self.limits,
                     "method": self.method,
                     "param_bounds": self.param_bounds,
                     "max_iters": self.max_iters,
                     "tol": self.tol}
        return json_dict

    def evaluate_metrics(self, trajectory, true_environment):
        """Evaluate all registered metrics.
        
        Args:
            trajectory (Trajectory): Final optimized trajectory
            true_environment (Environment): Ground truth environment
            
        Returns:
            dict: Dictionary of metric results
        """
        results = {}
        
        # Get the sampling distance from the reward if it's SampleValues
        samp_dist = 1.0  # Default sampling distance
        if hasattr(self.reward, 'params') and 'samp_dist' in self.reward.params:
            samp_dist = self.reward.params['samp_dist']
        print(f"reward hist: {self.reward_history}")
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            results[metric_name] = metric.evaluate(
                trajectory=trajectory,
                env_model=self.env_model,
                true_environment=true_environment,
                reward_history=self.reward_history,
                samp_dist=samp_dist,
                min_cost = self.min_cost
            )
        return results

    def _save_metric_results(self, results):
        """Save metric results to file and visualize them.
        
        Args:
            results (dict): Dictionary of metric results
        """
        # Create metrics directory if it doesn't exist
        metrics_path = os.path.join(self.path, "metrics")
        os.makedirs(metrics_path, exist_ok=True)
        
        # Save JSON results
        metric_json_path = os.path.join(metrics_path, f"metric_results_{self.id}.json")
        with open(metric_json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Create visualization
        metric_vis_path = os.path.join(metrics_path, f"metric_results_{self.id}.png")
        MetricVisualizer.plot_metric_history(results, metric_vis_path)
        
        print(f"Metrics saved to {metrics_path}")

    def get_plan(self, true_environment=None, soft_origin=None, soft_com=None, from_cache=False):
        """Get a plan by minimizing a cost funciton.

        Args:
            true_environment (Environment): Ground truth environment for metric evaluation
            soft_origin (tuple[float]): if not None, origin is encouraged to
                be near soft_origin with a soft constraint
            soft_com(tuple[float]): if not None, samples are encouraged to
                be far from soft_com with a soft constraint
            from_cache (bool): if True, uses the model cache
                during planning.
        """
        ###########################
        #### Setup constraints ####
        ###########################
        con = []
        bon = []
        if self.budget is not None:
            print("Adding budget constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += length_constraint(
                self.traj_generator.generate, budget=self.budget, method=self.method,constant_params=self.initial_params,scaling=self.scaling)

        if self.limits is not None:
            print("Adding safety boundary constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += bound_constraint(
                self.traj_generator.generate, limits=self.limits, method=self.method, constant_params=self.initial_params,scaling=self.scaling)

        if self.param_bounds is not None:
            print("Adding parameter bounds constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            param_bounds_subset = [self.param_bounds[i] for i in range(len(self.param_bounds)) if self.initial_params[i] is None]
            bon = param_constraint(param_bounds=param_bounds_subset, method=self.method,scaling=self.scaling)

        ##############################
        #### Add soft constraints ####
        ##############################

        def rew(theta):
            #print(f"Thetas entering rew: {theta[0]}, {theta[1]}")
            #if theta[0] < 0.05 or theta[1] < 0.05:
            #    print(f"\n\nerror: theta = {theta}\n\n") 
            return self.reward.eval(
                self.traj_generator.generate(*theta),
                self.env_model,
                from_cache=from_cache)

        if soft_origin is not None:
            print("Added soft origin constraint.")

        def s_origin(theta):
            if soft_origin is not None:
                # return soft_origin_penalty((theta[self.param_names['origin_x']],
                #                             theta[self.param_names['origin_y']]),
                #                            soft_origin)
                # [TODO] do we want a soft origin?
                return 0.0
            else:
                return 0.0

        if soft_com is not None:
            print("Added a diversity incentive.")

        def s_com(theta):
            if soft_com is not None:
                # return soft_origin_penalty((theta[self.param_names['origin_x']],
                #                             theta[self.param_names['origin_y']]),
                #                            soft_com)
                # [TODO] do we want a diversity bonus?
                return 0.0
            else:
                return 0.0

        def fun(theta):
            #print("function!")
            return rew(theta) + s_origin(theta) + s_com(theta)

        # Pass in scaled parameters, output reward on real parameters
        def make_scaled_fun(func_in,scaling):
            def scaled_fun(theta):
                scaled_theta = [val / scale for val,scale in zip(theta,scaling)]
                #print(f"scaled func! input {theta}, output {scaled_theta}")
                return func_in(scaled_theta) # Real world coordinates
            return scaled_fun

        def make_fun_subset(func_in,constant_params,scaling=[1,1,1,1,1]):
            def fun_subset(sub_theta):
                theta = []
                opt_var_count = 0
                #print(f"hi! fun subset")
                # Constant params should be without scaling
                for idx,param_value in enumerate(constant_params):
                    if param_value is not None:
                        theta += [param_value*scaling[idx]]
                    else:
                        theta += [sub_theta[opt_var_count]]
                        opt_var_count += 1
                #print(f"subset func! input {sub_theta}, output {theta}")
                return func_in(theta)
            return fun_subset

        ############################
        #### Solve optimization ####
        ############################

        # print(">>>>>Generator time.")
        # tic()
        # traj = self.traj_generator.generate(*self.x0)
        # toc()
        # tic()
        # test = self.reward.eval(traj, self.env_model, from_cache=from_cache)
        # toc()
        # print("<<<<<Done.")

        if self.max_iters is None:
            options = {'disp': True}
        else:
            options = {'disp': True, 'maxiter': self.max_iters}

        if self.tol is not None:
            if self.method == "SLSQP":
                options['ftol'] = self.tol
            elif self.method == "trust-constr":
                options['gtol'] = self.tol
                #options['verbose'] = 3
            elif self.method == "BFGS":
                options['xtol'] = self.tol

        #if self.method == "trust-constr":
        #    options['initial_tr_radius'] = 10 ######

        # Wrapper function that adjust input variable scale
        scaled_fun = make_scaled_fun(fun,self.scaling)
        print("Ready to optimize!")

        if self.method == "SLSQP" or \
                self.method == "trust-constr" or \
                self.method == "BFGS":

            #print(f"x0 for opt: {self.x0}")
            #print(f"method: {self.method}")
            #print(f"constraints: {con}")
            #for c in con:
                #if isinstance(c, optimize.NonlinearConstraint):
                    #print(f"Constraint: {c}, function = {c.fun}")
                    #constraint_value = c.fun(scale_vars(self.x0))  # Evaluate constraint function
                    #lb, ub = c.lb, c.ub  # Lower and upper bounds
                    #print(f"Nonlinear constraint at x0: {constraint_value}, should be in [{lb}, {ub}]")
                    
                    #if np.any(constraint_value < lb) or np.any(constraint_value > ub):
                    #    print("x0 violates this constraint!")
            if not self.hierarchy:
                print(f"Bounds: {bon}")
                res = optimize.minimize(
                    fun=scaled_fun,
                    jac=None,
                    x0=np.array([orig*scaled for orig,scaled in zip(self.x0,self.scaling)]),
                    options=options,
                    bounds=bon,
                    constraints=con,
                    method=self.method,
                    callback=self._callback)
            else:
                # Perform two different optimizations on subsets of the parameters
                # First optimization
                fun_subset = make_fun_subset(scaled_fun,self.initial_params,self.scaling)
                x0_subset = [self.x0[i]*self.scaling[i] for i in range(len(self.x0)) if self.initial_params[i] is None]
                #print(f"x0: {x0_subset}")
                res = optimize.minimize(
                    fun=fun_subset,
                    jac=None,
                    x0=np.array(x0_subset),
                    options=options,
                    bounds=bon, # should be bon
                    constraints=con,
                    method=self.method,
                    callback=self._callback)
                result1 = res.x
                print("Completed first optimization")

                # Second optimization
                new_x0_subset = []
                new_initial_params = []
                unpacked_results_count = 0 
                for index,value in enumerate(self.initial_params):
                    if value is None:
                        # Convert scaled to unscaled value
                        new_initial_params += [result1[unpacked_results_count] / self.scaling[index]]
                        unpacked_results_count += 1
                    else:
                        new_x0_subset += [self.x0[index] * self.scaling[index] * self.adj_secondary[index]]
                        new_initial_params += [None]
                self.secondary_params = new_initial_params
                fun_subset = make_fun_subset(scaled_fun,self.secondary_params,self.scaling)

                ### TODO: maybe break into separate function? if we're assembling constraints
                ### multiple times
                ### Reconstruct constraints
                ###########################
                #### Setup constraints ####
                ###########################
                con = []
                bon = []
                if self.budget is not None:
                    print("Adding budget constraint.")
                    if self.method != "SLSQP" and self.method != "trust-constr":
                        raise ValueError("Cannot perform constrained optimization.")

                    # Instantiate constraint
                    con += length_constraint(
                        self.traj_generator.generate, budget=self.budget, method=self.method,constant_params=new_initial_params,scaling=self.scaling)

                if self.limits is not None:
                    print("Adding safety boundary constraint.")
                    if self.method != "SLSQP" and self.method != "trust-constr":
                        raise ValueError("Cannot perform constrained optimization.")

                    # Instantiate constraint
                    con += bound_constraint(
                        self.traj_generator.generate, limits=self.limits, method=self.method,constant_params=new_initial_params,scaling=self.scaling)

                if self.param_bounds is not None:
                    print("Adding parameter bounds constraint.")
                    if self.method != "SLSQP" and self.method != "trust-constr":
                        raise ValueError("Cannot perform constrained optimization.")

                    # Instantiate constraint
                    param_bounds_subset = [self.param_bounds[i] for i in range(len(self.param_bounds)) if new_initial_params[i] is None]
                    bon = param_constraint(param_bounds=param_bounds_subset, method=self.method,scaling=self.scaling)

                res = optimize.minimize(
                    fun=fun_subset,
                    jac=None,
                    x0=np.array(new_x0_subset),
                    options=options,
                    bounds=bon, # should be bon
                    constraints=con,
                    method=self.method,
                    callback=self._callback_secondary)
                
                result2 = []
                result1_idx = 0
                result2_idx = 0
                for index,value in enumerate(self.initial_params):
                    print(f"Index:{index}, Value:{value}")
                    if value is None:
                        result2 += [result1[result1_idx]]
                        result1_idx += 1
                    else:
                        result2 += [res.x[result2_idx]]
                        result2_idx += 1
                print(result2)

        elif self.method == "basinhopping":
            res = optimize.basinhopping(
                func=lambda theta: self.reward.eval(
                    self.traj_generator.generate(*theta), self.env_model),
                x0=np.array(self.x0),
                constraints=con,
                disp=True,
                callback=self._callback)
        else:
            raise ValueError(f"Unrecognized optimization method {self.method}."
                             f"Should be one of SLSQP, trust-constr, BFGS, or basinhopping.")

        if not self.hierarchy:
            unscaled_x = self.unscale_vars(res.x)
            result_traj = self.traj_generator.generate(*unscaled_x)
            print("Optimization completed. Result:", unscaled_x)
            print("Length:", self.traj_generator.generate(*unscaled_x).length)
            # Evaluate metrics if provided and we have ground truth
            if self.metrics and true_environment is not None:
                print("Evaluating optimization metrics...")
                metric_results = self.evaluate_metrics(result_traj, true_environment)
                self._save_metric_results(metric_results)
            return result_traj
        else:
            unscaled_x = self.unscale_vars(result2)
            result_traj = self.traj_generator.generate(*unscaled_x)
            print("Optimization completed. Result:", unscaled_x)
            print("Length:", self.traj_generator.generate(*unscaled_x).length)
            # Evaluate metrics if provided and we have ground truth
            if self.metrics and true_environment is not None:
                print("Evaluating optimization metrics...")
                metric_results = self.evaluate_metrics(result_traj, true_environment)
                self._save_metric_results(metric_results)
            return result_traj

    def _callback(self, x, *args):
        """This callback is called during every iteration of optimization."""
        # rew = self.reward.eval(self.traj_generator.generate(*x), self.env_model)
        # import pdb; pdb.set_trace()
        const_params = self.constant_params
        x_old = x.copy()
        x = reconstruct_theta(const_params,self.unscale_vars(x))
        if x[0] <= 1e-3 or x[1] <= 1e-3:
            print("\n\n\nLENGTH OR HEIGHT IS NEGATIVE\n\n\n")
        print(f"callback! {x} (scaled from {x_old})")
        rew = args[0].fun
        self.check_min_cost(x,rew)
        self.reward_history.append(rew)
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.xlabel("Iterations")
        plt.ylabel("(Negative) Reward")
        plt.title("Optimization progress (should go down)")
        plt.savefig(os.path.join(self.path, f"training_progress_{self.id}plot.png"))
        plt.close()

        print(
            # f"n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            f"n:{self.neval}\t Value:{rew}")
            # f"n:{self.neval}")
        if not (self.neval % 5):
            print(
                f"\t n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            print("Saving mission checkpoint.")

            traj = self.traj_generator.generate(*x)
            visualize_and_save_traj(
                traj,
                self.env_model.extent,
                traj_name=os.path.join(self.path, f"temp_{self.id}iter{self.neval}"))
            print("Done.")
        self.neval += 1

    def _callback_secondary(self, x, *args):
        """This callback is called during every iteration of optimization."""
        # rew = self.reward.eval(self.traj_generator.generate(*x), self.env_model)
        # import pdb; pdb.set_trace()
        const_params = self.secondary_params
        x = reconstruct_theta(const_params,self.unscale_vars(x))
        print(f"callback! {x}")
        rew = args[0].fun
        self.check_min_cost(x,rew)
        self.reward_history.append(rew)
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.xlabel("Iterations")
        plt.ylabel("(Negative) Reward")
        plt.title("Optimization progress (should go down)")
        plt.savefig(os.path.join(self.path, f"training_progress_{self.id}plot.png"))
        plt.close()

        print(
            # f"n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            f"n:{self.neval}\t Value:{rew}")
            # f"n:{self.neval}")
        if not (self.neval % 5):
            print(
                f"\t n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            print("Saving mission checkpoint.")

            traj = self.traj_generator.generate(*x)
            visualize_and_save_traj(
                traj,
                self.env_model.extent,
                traj_name=os.path.join(self.path, f"temp_{self.id}iter{self.neval}"))
            print("Done.")
        self.neval += 1
    
    def unscale_vars(self,vars):
        """
        Translate scaled variables to unscaled variables
        """
        return [var/scale for var,scale in zip(vars,self.scaling)] 

    def scale_vars(self,vars):
        """
        Translate unscaled variables to scaled variables
        """
        return [var*scale for var,scale in zip(vars,self.scaling)] 

    def check_min_cost(self,x,cost):
        """
        Record current parameters if no lower value of the cost function is known
        """
        if self.min_cost[0] is None or self.min_cost[1] >= cost:
            self.min_cost[0] = x
            self.min_cost[1] = cost

