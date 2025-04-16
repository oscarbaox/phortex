import numpy as np
from scipy import optimize
from scipy.optimize import minimize

import os

import matplotlib.pyplot as plt

from fumes.utils import tic, toc
from fumes.simulator.utils import visualize_and_save_traj

from .planner import Planner
from .utils import length_constraint, bound_constraint, param_constraint, \
    soft_origin_penalty, reconstruct_theta


class TrajectoryOpt(Planner):
    def __init__(self, env_model, traj_generator, reward, x0, budget=None,
                 limits=[0., 1000., 0., 1000.], param_bounds=None,
                 param_names=None, max_iters=30, tol=1e-8,
                 method="trust-constr", experiment_name=None, hierarchy=False,
                 initial_params=[None,None,None,None,None]):
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

    def get_plan(self, soft_origin=None, soft_com=None, from_cache=False):
        """Get a plan by minimizing a cost funciton.

        Args:
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
                self.traj_generator.generate, budget=self.budget, method=self.method,constant_params=self.initial_params)

        if self.limits is not None:
            print("Adding safety boundary constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += bound_constraint(
                self.traj_generator.generate, limits=self.limits, method=self.method, constant_params=self.initial_params)

        if self.param_bounds is not None:
            print("Adding parameter bounds constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            param_bounds_subset = [self.param_bounds[i] for i in range(len(self.param_bounds)) if self.initial_params[i] is None]
            bon += param_constraint(param_bounds=param_bounds_subset, method=self.method)

        ##############################
        #### Add soft constraints ####
        ##############################

        def rew(theta):
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
            return rew(theta) + s_origin(theta) + s_com(theta)

        def make_fun_subset(constant_params):
            def fun_subset(sub_theta):
                theta = []
                opt_var_count = 0
                for param_value in constant_params:
                    if param_value is not None:
                        theta += [param_value]
                    else:
                        theta += [sub_theta[opt_var_count]]
                        opt_var_count += 1
                return fun(theta)
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
            elif self.method == "BFGS":
                options['xtol'] = self.tol

        if self.method == "SLSQP" or \
                self.method == "trust-constr" or \
                self.method == "BFGS":

            #print(f"x0 for opt: {self.x0}")
            #print(f"method: {self.method}")
            #print(f"constraints: {con}")
            for c in con:
                if isinstance(c, optimize.NonlinearConstraint):
                    #print(f"Constraint: {c}, function = {c.fun}")
                    constraint_value = c.fun(self.x0)  # Evaluate constraint function
                    lb, ub = c.lb, c.ub  # Lower and upper bounds
                    #print(f"Nonlinear constraint at x0: {constraint_value}, should be in [{lb}, {ub}]")
                    
                    #if np.any(constraint_value < lb) or np.any(constraint_value > ub):
                    #    print("x0 violates this constraint!")
            if not self.hierarchy:
                res = optimize.minimize(
                    fun=fun,
                    jac=None,
                    x0=np.array(self.x0),
                    options=options,
                    bounds=bon,
                    constraints=con,
                    method=self.method,
                    callback=self._callback)
            else:
                # Perform two different optimizations on subsets of the parameters
                # First optimization
                fun_subset = make_fun_subset(self.initial_params)
                x0_subset = [self.x0[i] for i in range(len(self.x0)) if self.initial_params[i] is None]
                res = optimize.minimize(
                    fun=fun_subset,
                    jac=None,
                    x0=np.array(x0_subset),
                    options=options,
                    bounds=bon,
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
                        new_initial_params += [result1[unpacked_results_count]]
                        unpacked_results_count += 1
                    else:
                        new_x0_subset += [self.x0[index]]
                        new_initial_params += [None]
                self.secondary_params = new_initial_params
                fun_subset = make_fun_subset(self.secondary_params)

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
                        self.traj_generator.generate, budget=self.budget, method=self.method,constant_params=new_initial_params)

                if self.limits is not None:
                    print("Adding safety boundary constraint.")
                    if self.method != "SLSQP" and self.method != "trust-constr":
                        raise ValueError("Cannot perform constrained optimization.")

                    # Instantiate constraint
                    con += bound_constraint(
                        self.traj_generator.generate, limits=self.limits, method=self.method,constant_params=new_initial_params)

                if self.param_bounds is not None:
                    print("Adding parameter bounds constraint.")
                    if self.method != "SLSQP" and self.method != "trust-constr":
                        raise ValueError("Cannot perform constrained optimization.")

                    # Instantiate constraint
                    param_bounds_subset = [self.param_bounds[i] for i in range(len(self.param_bounds)) if new_initial_params[i] is None]
                    bon += param_constraint(param_bounds=param_bounds_subset, method=self.method)

                res = optimize.minimize(
                    fun=fun_subset,
                    jac=None,
                    x0=np.array(new_x0_subset),
                    options=options,
                    bounds=bon,
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
            print("Optimization completed. Result:", res.x)
            print("Length:", self.traj_generator.generate(*res.x).length)
            return self.traj_generator.generate(*res.x)
        else:
            print("Optimization completed. Result:", result2)
            print("Length:", self.traj_generator.generate(*result2).length)
            return self.traj_generator.generate(*result2)

    def _callback(self, x, *args):
        """This callback is called during every iteration of optimization."""
        # rew = self.reward.eval(self.traj_generator.generate(*x), self.env_model)
        # import pdb; pdb.set_trace()
        const_params = self.constant_params
        x = reconstruct_theta(const_params,x)
        rew = args[0].fun
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
        if not (self.neval % 10):
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
        x = reconstruct_theta(const_params,x)
        rew = args[0].fun
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
        if not (self.neval % 10):
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

