from time import sleep

import gcsfs
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from sklearn.neighbors import NearestNeighbors
from tqdm_joblib import tqdm_joblib

from connectomics.segclr import reader

from ..base import BaseQuery


def _format_l2_data(level2_data: dict) -> pd.DataFrame:
    level2_data = pd.DataFrame(level2_data).T

    # TODO handle empty cache better
    # TODO just make this happen all at once with a single apply/
    level2_data["x"] = level2_data["rep_coord_nm"].apply(
        lambda x: x[0] if x is not None else None
    )
    level2_data["y"] = level2_data["rep_coord_nm"].apply(
        lambda x: x[1] if x is not None else None
    )
    level2_data["z"] = level2_data["rep_coord_nm"].apply(
        lambda x: x[2] if x is not None else None
    )

    level2_data.index.name = "level2_id"
    level2_data.index = level2_data.index.astype(int)
    return level2_data


class SegCLRQuery(BaseQuery):
    def __init__(self, client, *args, **kwargs):
        super().__init__(client, *args, **kwargs)
        self.version = 343
        self.timestamp = client.materialize.get_timestamp(self.version)

    def set_query_ids(self, query_ids: ArrayLike):
        # TODO input validation
        query_ids = np.unique(query_ids)
        self.query_ids = query_ids

    def map_to_version(self):
        def _get_backward_id_map_for_chunk(chunk):
            out = self.client.chunkedgraph.get_past_ids(
                chunk, timestamp_past=self.timestamp
            )
            return out["past_id_map"]

        chunk_size = 1000
        root_ids = self.query_ids
        n_chunks = len(root_ids) // chunk_size + 1
        chunks = np.array_split(root_ids, n_chunks)

        with tqdm_joblib(
            desc="Getting IDs at SegCLR version",
            total=n_chunks,
            disable=self.verbose < 1,
        ):
            backward_id_maps = Parallel(n_jobs=self.n_jobs)(
                delayed(_get_backward_id_map_for_chunk)(chunk) for chunk in chunks
            )

        backward_id_map = {}
        for past_id_map_chunk in backward_id_maps:
            backward_id_map.update(past_id_map_chunk)

        forward_id_map = {}
        for current_id, past_ids in backward_id_map.items():
            for past_id in past_ids:
                forward_id_map[past_id] = current_id

        versioned_query_ids = np.unique(list(forward_id_map.keys()))

        self.versioned_query_ids_ = versioned_query_ids
        self.forward_id_map_ = forward_id_map
        self.backward_id_map_ = backward_id_map

    def get_embeddings(self):
        # TODO these are hard-coded for now
        PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")
        embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)

        # TODO do this in a smarter way to look up shards for every versioned_id first
        # and then group the lookups by shard
        def _get_embeddings_for_versioned_id(past_id):
            past_id = int(past_id)
            root_id = self.forward_id_map_[past_id]
            try:
                out = embedding_reader[past_id]
                new_out = {}
                for xyz, embedding_vector in out.items():
                    new_out[(root_id, past_id, *xyz)] = embedding_vector
            except KeyError:
                new_out = {}
            return new_out

        versioned_query_ids = self.versioned_query_ids_
        with tqdm_joblib(
            desc="Getting SegCLR embeddings",
            total=len(versioned_query_ids),
            disable=self.verbose < 1,
        ):
            embeddings_dicts = Parallel(n_jobs=self.n_jobs)(
                delayed(_get_embeddings_for_versioned_id)(past_id)
                for past_id in versioned_query_ids
            )

        embeddings_dict = {}
        for d in embeddings_dicts:
            embeddings_dict.update(d)

        embedding_df = pd.DataFrame(embeddings_dict).T
        embedding_df.index.names = ["current_id", "versioned_id", "x", "y", "z"]

        embedding_df["x_nm"] = embedding_df.index.get_level_values("x") * 32
        embedding_df["y_nm"] = embedding_df.index.get_level_values("y") * 32
        embedding_df["z_nm"] = embedding_df.index.get_level_values("z") * 40
        embedding_df = embedding_df.droplevel(["x", "y", "z"])
        embedding_df.rename(
            columns={"x_nm": "x", "y_nm": "y", "z_nm": "z"}, inplace=True
        )
        mystery_offset = np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
        embedding_df["x"] += mystery_offset[0]
        embedding_df["y"] += mystery_offset[1]
        embedding_df["z"] += mystery_offset[2]
        embedding_df["x"] = embedding_df["x"].astype(int)
        embedding_df["y"] = embedding_df["y"].astype(int)
        embedding_df["z"] = embedding_df["z"].astype(int)

        embedding_df.set_index(["x", "y", "z"], append=True, inplace=True)

        self.features_ = embedding_df

    def map_to_level2(self):
        def _map_to_level2_for_root(root_id, root_data, n_tries=3):
            try:
                level2_ids = self.client.chunkedgraph.get_leaves(root_id, stop_layer=2)
                level2_data = self.client.l2cache.get_l2data(level2_ids)
                level2_data = _format_l2_data(level2_data)

                nn = NearestNeighbors(n_neighbors=1)
                nn.fit(level2_data[["x", "y", "z"]].values)
                distances, indices = nn.kneighbors(
                    root_data.reset_index()[["x", "y", "z"]].values
                )
                distances = distances.flatten()
                indices = indices.flatten()

                info_df = pd.DataFrame(index=root_data.index)
                info_df["level2_id"] = level2_data.index[indices]
                info_df["distance_to_level2_node"] = distances
            except Exception as e:
                if n_tries > 0:
                    sleep(self.wait_time)
                    return _map_to_level2_for_root(
                        root_id, root_data, n_tries=n_tries - 1
                    )
                else:
                    raise e
            return info_df

        with tqdm_joblib(
            desc="Mapping to level2 nodes",
            total=len(self.features_.index.get_level_values("current_id").unique()),
            disable=self.verbose < 1,
        ):
            info_dfs = Parallel(n_jobs=-1)(
                delayed(_map_to_level2_for_root)(root_id, root_data)
                for root_id, root_data in self.features_.groupby("current_id")
            )

        info_df = pd.concat(info_dfs)
        self.level2_mapping_ = info_df
