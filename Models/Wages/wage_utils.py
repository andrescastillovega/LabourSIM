# Import libraries
import pandas as pd
import numpy as np
import re
from io import StringIO 
import sys


class Capturing(list):
    """Create a context manager that captures stdout output (Numpyro prints to stdout)"""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def sampling_output(sampling_info, nchains, ndraws, ntunes):
    """ Function to organize the sampling info after the NUTS sampler has run."""
    sampling_metadata = {}
    for location, stage in enumerate(["Compilation", "Sampling", "Transformation", "Log Likelihood"]):
        stage_duration = re.search(fr'{stage} time =\s+(\d+:\d+:\d+\.\d+)', sampling_info[(location * 2) + 1]).group(1)
        sampling_metadata[f"{stage}_time"] = stage_duration
        if stage == "Sampling":
            sampling_time_seconds = pd.to_timedelta(stage_duration).total_seconds()
            sampling_metadata["AvgIt/s"] = round(((ndraws + ntunes) * nchains) / sampling_time_seconds, 2)

    return sampling_metadata