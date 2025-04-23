""" Script to iterate over multiple options of the ybw_phumes_phortex example """
from fumes.examples.epr_examples import ybw_phumes_phortex

heights = [10, 20, 50, 100]
lengths = [10, 25, 70, 200]
orders = [[0,0,0,0,0],[1,1,1,0,0],[1,1,0,0,0],[1,0,0,0,0]]
for order in orders:
    for h in heights:
        for l in lengths:
            params = {
                "height":h,
                "length":l,
                "order":order
            }
            ybw_phumes_phortex.main_with_params(params)
            print("Ran sim!")


 