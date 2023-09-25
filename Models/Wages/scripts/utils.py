import pandas as pd
import arviz as az
import os

def get_rhat_max(trace):
    summary = az.summary(trace)
    rhat_max = summary["r_hat"].max()
    return rhat_max

def save_summary(trace, model_name, year=None):
    if os.path.exists("../outputs/compilate_summary.csv"):
        compilate_summary = pd.read_csv("../outputs/compilate_summary.csv")
    else:
        compilate_summary = pd.DataFrame()
    summary = az.summary(trace)

    summary["model"] = model_name
    if year is None:
        summary["year"] = "all"
    else:
        summary["year"] = year

    compilate_summary = pd.concat([compilate_summary, summary])
    compilate_summary.to_csv("../outputs/compilate_summary.csv", index=True)

    
    