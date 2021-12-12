import numpy as np
import flaml
import ray

Search_Alg = "CFOCat"

if Search_Alg  == "BlendSearch":
    from flaml import tune
else:
    from ray import tune

def test_func(config: dict):
    
    funcLoss = 50
    for key, value in config.items():
        if key in ["x1","x2","x3","x4","x5"]:
            funcLoss+= value**2 - 10 * np.cos(2*np.pi*value)

    if "incumbent_info" in config.keys():
        print("incumbent_result",config["incumbent_info"]["incumbent_result"])

    tune.report(funcLoss = funcLoss)

def test_record_incumbent(method="BlendSearch"):

    if method != "CFOCat":
        search_space = {
            "x1": tune.randint(1, 9),
            "x2": tune.randint(1, 9),
            "x3": tune.randint(1, 9),
            "x4": tune.randint(1, 9),
            "x5": tune.randint(1, 9),
        }
    else:
        search_space = {
            "x1": tune.choice([1,2,3,4,5,6,7,8,9,10]),
            "x2": tune.choice([1,2,3,4,5,6,7,8,9,10]),
            "x3": tune.choice([1,2,3,4,5,6,7,8,9,10]),
            "x4": tune.choice([1,2,3,4,5,6,7,8,9,10]),
            "x5": tune.choice([1,2,3,4,5,6,7,8,9,10]),
        }


    max_iter = 20
    num_samples = 128
    time_budget_s = 60
    n_cpu = 4
    ray.shutdown()
    ray.init(num_cpus=n_cpu, num_gpus=0)

    if method == "BlendSearch":
        tune.run(
            test_func,
            config=search_space,
            metric="funcLoss",
            mode="min",
            max_resource=max_iter,
            min_resource=1,
            resources_per_trial={"cpu": 1},
            local_dir="logs/",
            num_samples=num_samples * n_cpu,
            time_budget_s=time_budget_s,
            use_ray= True,
            use_incumbent_result= True,
        )
        return 
    elif method == "CFO":
        from flaml import CFO
        algo = CFO(
            low_cost_partial_config={
                "max_depth": 1,
            },
            cat_hp_cost={
                "min_child_weight": [6, 3, 2],
            },
            use_incumbent_result= True,
        )
    elif method == "CFOCat":
        from flaml.searcher.cfo_cat import CFOCat
        algo = CFOCat(
            use_incumbent_result= True,
        )
    else:
        raise NotImplementedError
    tune.run(
        test_func,
        metric="funcLoss",
        mode="min",
        resources_per_trial={"cpu": 1},
        config=search_space,
        local_dir="logs/",
        num_samples=num_samples * n_cpu,
        time_budget_s=time_budget_s,
        search_alg=algo,
    )

if __name__ == "__main__":
    test_record_incumbent(method = Search_Alg)
