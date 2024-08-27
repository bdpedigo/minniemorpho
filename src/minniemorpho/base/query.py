from caveclient.frameworkclient import CAVEclientFull
from typing import Optional

class BaseQuery:
    def __init__(
        self,
        client: CAVEclientFull,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        *args,
        **kwargs,
    ):
        self.client = client
        self.verbose = verbose
        self.n_jobs = n_jobs
