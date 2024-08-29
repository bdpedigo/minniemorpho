from typing import Optional

from caveclient.frameworkclient import CAVEclientFull


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
