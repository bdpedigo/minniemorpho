from itertools import chain
from time import sleep
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from tqdm_joblib import tqdm_joblib

from ..base import BaseQuery


def _format_l2_data(level2_data: dict) -> pd.DataFrame:
    level2_data = pd.DataFrame(level2_data).T
    if not level2_data.empty:
        # TODO handle empty cache better
        # TODO just make this happen all at once with a single apply/
        level2_data["x"] = level2_data["rep_coord_nm"].apply(
            lambda x: x[0] if x is not None else None
        ).astype(int)
        level2_data["y"] = level2_data["rep_coord_nm"].apply(
            lambda x: x[1] if x is not None else None
        ).astype(int)
        level2_data["z"] = level2_data["rep_coord_nm"].apply(
            lambda x: x[2] if x is not None else None
        ).astype(int)
        level2_data = level2_data.drop(columns=["rep_coord_nm"])

        level2_data.index.name = "level2_id"
        level2_data.index = level2_data.index.astype(int)
    return level2_data


class Level2Query(BaseQuery):
    def __init__(self, *args, attributes: Optional[list] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.attributes = attributes

    def _get_level2_nodes(self):
        def _get_level2_for_root(root_id, n_tries=3):
            try:
                if hasattr(self, "bounds_seg"):
                    bounds = self.bounds_seg.T
                else:
                    bounds = None
                level2_ids = self.client.chunkedgraph.get_leaves(
                    root_id, stop_layer=2, bounds=bounds
                )
            except Exception as e:  # TODO make this specific to server errors
                if n_tries > 0:
                    sleep(self.wait_time)
                    return _get_level2_for_root(root_id, n_tries=n_tries - 1)
                else:
                    if self.continue_on_error:
                        level2_ids = []
                    else:
                        raise e
            return level2_ids

        with tqdm_joblib(
            desc="Getting level2 nodes",
            total=len(self.query_ids),
            disable=self.verbose < 1,
        ):
            level2_ids_by_root = Parallel(n_jobs=self.n_jobs)(
                delayed(_get_level2_for_root)(root_id) for root_id in self.query_ids
            )

        roots_broadcast = [
            [root_id] * len(level2_ids)
            for root_id, level2_ids in zip(self.query_ids, level2_ids_by_root)
        ]
        roots_broadcast = list(chain.from_iterable(roots_broadcast))
        level2_ids = list(chain.from_iterable(level2_ids_by_root))

        root_level2_mapping = pd.DataFrame(
            {"root_id": roots_broadcast, "level2_id": level2_ids}
        )
        assert root_level2_mapping["level2_id"].is_unique

        self.root_level2_mapping_ = root_level2_mapping

    def _get_level2_data(self, chunk_size=10_000):
        def _get_level2_data_for_chunk(level2_ids, n_tries=3):
            try:
                level2_data = self.client.l2cache.get_l2data(
                    level2_ids, attributes=self.attributes
                )
                level2_data = _format_l2_data(level2_data)
            except Exception as e:
                if n_tries > 0:
                    sleep(self.wait_time)
                    return _get_level2_data_for_chunk(level2_ids, n_tries=n_tries - 1)
                else:
                    if self.continue_on_error:
                        level2_data = pd.DataFrame()
                    else:
                        raise e

            return level2_data

        level2_ids = self.root_level2_mapping_["level2_id"].unique()
        n_chunks = np.ceil(len(level2_ids) / chunk_size).astype(int)
        level2_id_chunks = np.array_split(level2_ids, n_chunks)

        with tqdm_joblib(
            desc="Getting level2 data",
            total=len(level2_id_chunks),
            disable=self.verbose < 1,
        ):
            chunked_level2_data = Parallel(n_jobs=self.n_jobs)(
                delayed(_get_level2_data_for_chunk)(chunk) for chunk in level2_id_chunks
            )

        chunked_level2_data = [df for df in chunked_level2_data if not df.empty]
        level2_data = pd.concat(chunked_level2_data)

        level2_data["root_id"] = level2_data.index.map(
            self.root_level2_mapping_.set_index("level2_id")["root_id"]
        )

        level2_data = level2_data.reset_index().set_index(["root_id", "level2_id"])

        self.features_ = level2_data

    def get_features(self):
        self._get_level2_nodes()
        self._get_level2_data()

    # def _old_get_features(self):
    #     def _map_to_level2_for_root(root_id, root_data, n_tries=3):
    #         try:
    #             if hasattr(self, "bounds_seg"):
    #                 bounds = self.bounds_seg.T
    #             else:
    #                 bounds = None
    #             level2_ids = self.client.chunkedgraph.get_leaves(
    #                 root_id, stop_layer=2, bounds=bounds
    #             )
    #             level2_data = self.client.l2cache.get_l2data(
    #                 level2_ids, attributes=["rep_coord_nm", "size_nm3", "area_nm2"]
    #             )
    #             level2_data = _format_l2_data(level2_data)
    #             if not level2_data.empty:
    #                 nn = NearestNeighbors(n_neighbors=1)
    #                 nn.fit(level2_data[["x", "y", "z"]].values)
    #                 distances, indices = nn.kneighbors(
    #                     root_data.reset_index()[["x", "y", "z"]].values
    #                 )
    #                 distances = distances.flatten()
    #                 indices = indices.flatten()

    #                 info_df = pd.DataFrame(index=root_data.index)
    #                 info_df["level2_id"] = level2_data.index[indices]
    #                 info_df["distance_to_level2_node"] = distances
    #             else:
    #                 info_df = pd.DataFrame(
    #                     index=root_data.index,
    #                     columns=["level2_id", "distance_to_level2_node"],
    #                 )
    #         except Exception as e:  # TODO make this specific to server errors
    #             if n_tries > 0:
    #                 sleep(self.wait_time)
    #                 return _map_to_level2_for_root(
    #                     root_id, root_data, n_tries=n_tries - 1
    #                 )
    #             else:
    #                 if self.continue_on_error:
    #                     info_df = pd.DataFrame(
    #                         index=root_data.index,
    #                         columns=["level2_id", "distance_to_level2_node"],
    #                     )
    #                 else:
    #                     raise e
    #         return info_df

    #     with tqdm_joblib(
    #         desc="Mapping to level2 nodes",
    #         total=len(self.features_.index.get_level_values("current_id").unique()),
    #         disable=self.verbose < 1,
    #     ):
    #         info_dfs = Parallel(n_jobs=-1)(
    #             delayed(_map_to_level2_for_root)(root_id, root_data)
    #             for root_id, root_data in self.features_.groupby("current_id")
    #         )

    #     info_df = pd.concat(info_dfs)
    #     self.level2_mapping_ = info_df
