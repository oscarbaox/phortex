""" Script to iterate over multiple options of the ybw_phumes_phortex example """
from fumes.examples.epr_examples import ybw_phumes_phortex

rewards = ["SampleValuesPrioritizeMid"]
count = 0
failed = 0
#order_scale_groups = [([0,0,0,0,0],[1,1,1,1,1]),([0,0,0,0,0],[0.5,0.5,1,1,1]),
#                      ([0,0,0,0,0],[0.1,0.1,1,1,1]),([1,1,1,0,0],[1,1,1,1,1]),
#                      ([1,1,1,0,0],[0.5,0.5,1,1,1])]

candidate_params = [[50,100,1,[0,0,0,0,0]],[25,50,1,[0,0,0,0,0]],
                    [10,10,1,[0,0,0,0,0]],[10,100,1,[1,1,1,0,0]],
                    [50,100,1,[1,1,1,0,0]],[10,50,0.5,[1,1,1,0,0]],
                    [50,10,0.5,[1,1,1,0,0]],[50,50,0.5,[0,0,0,0,0]],
                    [50,10,0.5,[0,0,0,0,0]],[10,50,0.1,[0,0,0,0,0]]]

for reward in rewards:
    for i in range(5):
        for group in candidate_params:
            try:
                params = {
                    "height":group[0],
                    "length":group[1],
                    "order":group[3],
                    "scaling":[group[2],group[2],1,1,1],
                    "adjust_secondary_params":[1,1,1,1,1],
                    "reward_function":"SampleValuesPrioritizeMid",
                    "date":"0507",
                    "sim_num":i
                }
                ybw_phumes_phortex.main_with_params(params)
                count += 1
                print(f"Ran sim! Count: {count}, failed: {failed}, cycle {i}")
            except Exception:
                print(f"\n\n\nFailed for params = {params}")
                failed += 1


    