import pandas as pd
import arviz as az
import os

def get_rhat_max(trace):
    summary = az.summary(trace)
    rhat_max = summary["r_hat"].max()
    return rhat_max

def save_summary(workflow, trace, model_name, year=None):
    if os.path.exists(f"../outputs/compilate_summary_{workflow}.csv"):
        compilate_summary = pd.read_csv(f"../outputs/compilate_summary_{workflow}.csv")
    else:
        compilate_summary = pd.DataFrame()
    summary = az.summary(trace)

    summary["model"] = model_name
    if year is None:
        summary["year"] = "all"
        summary.to_csv(f"../outputs/{model_name}/summary.csv", index=True)
    else:
        summary["year"] = year
        summary.to_csv(f"../outputs/{model_name}/{year}/summary.csv", index=True)

    compilate_summary = pd.concat([compilate_summary, summary])
    compilate_summary.to_csv(f"../outputs/{model_name}/compilate_summary_{workflow}.csv", index=True)

def check_workflow(workflow):
    for model in workflow:
        print(list(model.keys())[0], ",".join([param for param in list(model.values())[0]["parameters"].keys()]))



    
    