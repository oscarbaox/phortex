""" Script to iterate over multiple options of the ybw_phumes_phortex example """
from fumes.examples.epr_examples import ybw_phumes_phortex

heights = [10, 20, 50, 100]
lengths = [10, 25, 70, 200]
orders = [[1,1,0,0,0],[1,1,1,0,0],[0,0,0,0,0]]
rect_scaling = [0.01, 0.1, 0.5, 1]
secondary_scaling = [1, 5, 10]
reward = "SampleValuesPrioritizeMid"
for i in range(4):
    for order in orders:
        for rect_scale in rect_scaling:
            try:
                params = {
                    "height":heights[i],
                    "length":lengths[i],
                    "order":order,
                    "scaling":[rect_scale,rect_scale,1,1,1],
                    "adjust_secondary_params":[1,1,1,1,1],
                    "reward_function":"SampleValuesPrioritizeMid"
                }
                ybw_phumes_phortex.main_with_params(params)
                print("Ran sim!")
            except Exception:
                print(f"\n\n\nFailed for params = {params}")


    