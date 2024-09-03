from typing import Optional

import numpy as np
from caveclient.frameworkclient import CAVEclientFull
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm_joblib import tqdm_joblib


class BaseQuery:
    def __init__(
        self,
        client: CAVEclientFull,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        continue_on_error: bool = False,
        *args,
        **kwargs,
    ):
        self.client = client
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.wait_time = 0.5
        self.continue_on_error = continue_on_error

    def set_query_ids(self, query_ids: ArrayLike):
        # TODO input validation
        query_ids = np.unique(query_ids)
        self.query_ids = query_ids

    def set_query_bounds(self, bounds: ArrayLike):
        assert bounds.shape == (2, 3)
        self.bounds = bounds
        self.bounds_seg = (bounds / self.client.chunkedgraph.base_resolution).astype(
            int
        )

    def _crop_to_bounds(self, dataframe):
        if not hasattr(self, "bounds"):
            return dataframe
        x_min, y_min, z_min = self.bounds[0]
        x_max, y_max, z_max = self.bounds[1]
        query_str = "x >= @x_min & x <= @x_max"
        query_str += " & y >= @y_min & y <= @y_max"
        query_str += " & z >= @z_min & z <= @z_max"
        return dataframe.query(query_str)

    # def _dispatch_parallel(self, method, *args, desc=None, total=None, **kwargs):
    #     with tqdm_joblib(
    #         desc=desc,
    #         total=total,
    #         disable=self.verbose < 1,
    #     ):
    #         out = Parallel(n_jobs=self.n_jobs)(
    #             delayed(method)(arg, **kwargs) for arg in args
    #         )

    #     return out
