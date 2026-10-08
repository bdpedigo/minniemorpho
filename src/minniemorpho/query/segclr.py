import io
import os
import struct
import zipfile
from typing import Optional

import gcsfs
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

from connectomics.common import sharding
from connectomics.segclr.reader import DATA_URL_FROM_KEY_BYTEWIDTH64, EmbeddingReader

from .base import BaseQuery

DATA_URL_FROM_KEY_BYTEWIDTH8 = {
    "microns_v943": "gs://iarpa_microns/minnie/minnie65/embeddings_m943/segclr_nm_coord_public_offset_csvzips"
}


def get_reader(key: str, filesystem, num_shards: Optional[int] = None):
    """Convenience helper to get reader for given dataset key."""
    if key in DATA_URL_FROM_KEY_BYTEWIDTH64:
        url = DATA_URL_FROM_KEY_BYTEWIDTH64[key]
        bytewidth = 64
        num_shards = 10_000
    elif key in DATA_URL_FROM_KEY_BYTEWIDTH8:
        url = DATA_URL_FROM_KEY_BYTEWIDTH8[key]
        bytewidth = 8
        num_shards = 50_000
    else:
        raise ValueError(f"Key not found: {key}")

    def sharder(segment_id: int) -> int:
        return sharding.md5_shard(
            segment_id, num_shards=num_shards, bytewidth=bytewidth
        )

    return EmbeddingReader(filesystem, url, sharder)


def _format_embedding_343(embedding: dict) -> pd.DataFrame:
    embedding_df = pd.DataFrame(embedding).T
    embedding_df.index.names = ["root_id", "versioned_id", "x", "y", "z"]
    embedding_df["x_nm"] = embedding_df.index.get_level_values("x") * 32
    embedding_df["y_nm"] = embedding_df.index.get_level_values("y") * 32
    embedding_df["z_nm"] = embedding_df.index.get_level_values("z") * 40
    embedding_df = embedding_df.droplevel(["x", "y", "z"])
    embedding_df.rename(columns={"x_nm": "x", "y_nm": "y", "z_nm": "z"}, inplace=True)
    mystery_offset = np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
    embedding_df["x"] += mystery_offset[0]
    embedding_df["y"] += mystery_offset[1]
    embedding_df["z"] += mystery_offset[2]
    embedding_df["x"] = embedding_df["x"].astype(int)
    embedding_df["y"] = embedding_df["y"].astype(int)
    embedding_df["z"] = embedding_df["z"].astype(int)

    embedding_df.set_index(["x", "y", "z"], append=True, inplace=True)
    return embedding_df


def _format_embedding(embedding: dict) -> pd.DataFrame:
    embedding_df = pd.DataFrame(embedding).T
    embedding_df.index.names = ["root_id", "versioned_id", "x", "y", "z"]
    embedding_df = embedding_df.reset_index(level=["x", "y", "z"], drop=False)
    embedding_df["x"] = embedding_df["x"].astype(int)
    embedding_df["y"] = embedding_df["y"].astype(int)
    embedding_df["z"] = embedding_df["z"].astype(int)
    embedding_df.set_index(["x", "y", "z"], append=True, inplace=True)
    return embedding_df


def _select_components(embeddings: np.ndarray, components) -> np.ndarray:
    if components is None:
        return embeddings
    elif isinstance(components, int):
        return embeddings[:, :components]
    elif isinstance(components, slice):
        return embeddings[:, components]
    elif isinstance(components, (list, tuple)):
        return embeddings[:, components[0] : components[1]]
    else:
        raise ValueError(f"Invalid type for components : {type(components)}")


def _read_embeddings_in_bounds(
    embedding_reader: EmbeddingReader,
    root_id: int,
    seg_id: int,
    bounds: Optional[np.ndarray] = None,
    components=None,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Read one segment's CSV from its shard, cropping each chunk to bounds.

    Gives the same values as `EmbeddingReader[seg_id]` -> `_format_embedding` ->
    `_crop_to_bounds` (stored as float32), but holds at most the compressed member,
    never the full (possibly >1 GB) CSV or its parsed Python objects. Raises KeyError
    if the segment is not found.
    """
    shard = embedding_reader._sharder(seg_id)
    zip_path = os.path.join(embedding_reader._zipdir, f"{shard}.zip")
    fs = embedding_reader._filesystem
    with fs.open(zip_path) as f:
        with zipfile.ZipFile(f) as z:
            info = z.getinfo(f"{seg_id}.csv")
        file_size = f.size

    # fetch the whole compressed member in one ranged request: streaming it through
    # the file object instead does many sequential small fetches and is ~2x slower.
    # the local header's extra field can differ from the central directory's, so
    # over-fetch a little and fetch any remainder if that was not enough
    start = info.header_offset
    end = min(start + 30 + len(info.filename) + 65_536 + info.compress_size, file_size)
    member = fs.cat_file(zip_path, start=start, end=end)
    name_length, extra_length = struct.unpack("<HH", member[26:30])
    data_start = 30 + name_length + extra_length
    if data_start + info.compress_size > len(member):
        member += fs.cat_file(
            zip_path,
            start=start + len(member),
            end=start + data_start + info.compress_size,
        )

    kept = []
    buffer = io.BytesIO(member)
    buffer.seek(data_start)
    # decompress and parse in chunks so that the full CSV never sits in memory
    with zipfile.ZipExtFile(buffer, "r", info) as c:
        # "high" is pure C; "round_trip" takes the GIL per value and is ~10x slower
        # with many threads
        for chunk in pd.read_csv(
            c, header=None, chunksize=chunksize, float_precision="high"
        ):
            # columns are node_id, x, y, z, embedding...
            xyz = chunk.iloc[:, 1:4].to_numpy(dtype=np.float64).astype(int)
            if bounds is not None:
                mask = np.all((xyz >= bounds[0]) & (xyz <= bounds[1]), axis=1)
                chunk = chunk[mask]
                xyz = xyz[mask]
            if len(chunk) == 0:
                continue
            embeddings = chunk.iloc[:, 4:].to_numpy(dtype=np.float64)
            embeddings = _select_components(embeddings, components)
            # the released embeddings are float32 values: "high" can be 1 ulp off
            # in float64, and rounding to float32 recovers them exactly
            embeddings = embeddings.astype(np.float32)
            kept.append((xyz, embeddings))
    del buffer, member

    if kept:
        xyz = np.concatenate([k[0] for k in kept])
        embeddings = np.concatenate([k[1] for k in kept])
    else:
        n_cols = _select_components(np.empty((0, 0)), components).shape[1]
        xyz = np.empty((0, 3), dtype=int)
        embeddings = np.empty((0, n_cols), dtype=np.float32)

    n = len(xyz)
    index = pd.MultiIndex.from_arrays(
        [
            np.full(n, root_id, dtype=np.int64),
            np.full(n, seg_id, dtype=np.int64),
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
        ],
        names=["root_id", "versioned_id", "x", "y", "z"],
    )
    # object dtype for the column labels matches what `_format_embedding` gives
    columns = pd.Index(list(range(embeddings.shape[1])), dtype=object)
    return pd.DataFrame(embeddings, index=index, columns=columns)


class SegCLRQuery(BaseQuery):
    def __init__(self, client, *args, version: int = 943, components=None, **kwargs):
        super().__init__(client, *args, **kwargs)
        self.version = version
        self.timestamp = client.materialize.get_timestamp(self.version)
        self.components = components

    def _map_to_version(self):
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
            # threading backend: this is network I/O, not CPU work, so threads
            # avoid per-process memory duplication that processes would incur
            backward_id_maps = Parallel(n_jobs=self.n_jobs, backend="threading")(
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

    def _get_reader(self):
        PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")
        return get_reader(f"microns_v{self.version}", PUBLIC_GCSFS)

    def get_features(self):
        self._map_to_version()

        # TODO these are hard-coded for now

        embedding_reader = self._get_reader()

        # TODO do this in a smarter way to look up shards for every versioned_id first
        # and then group the lookups by shard
        def _get_embeddings_for_versioned_id(past_id) -> Optional[pd.DataFrame]:
            past_id = int(past_id)
            root_id = self.forward_id_map_[past_id]
            try:
                # stream and crop per chunk: whole-segment CSVs can exceed 1 GB,
                # and parsing them in full before cropping caused OOMs
                new_out = _read_embeddings_in_bounds(
                    embedding_reader,
                    root_id,
                    past_id,
                    bounds=getattr(self, "bounds", None),
                    components=self.components,
                )
            except KeyError:
                new_out = None
            return new_out

        versioned_query_ids = self.versioned_query_ids_
        with tqdm_joblib(
            desc="Getting SegCLR embeddings",
            total=len(versioned_query_ids),
            disable=self.verbose < 1,
        ):
            # threading backend: this is network I/O, not CPU work, so threads
            # avoid per-process memory duplication that processes would incur
            embedding_dfs = Parallel(
                n_jobs=self.n_jobs, timeout=99999, backend="threading"
            )(
                delayed(_get_embeddings_for_versioned_id)(past_id)
                for past_id in versioned_query_ids
            )

        embedding_dfs = [df for df in embedding_dfs if df is not None]
        embedding_df = pd.concat(embedding_dfs, axis=0)

        # embeddings_dict = {}
        # for d in embeddings_dicts:
        #     embeddings_dict.update(d)

        # embedding_df = pd.DataFrame(embeddings_dict).T
        # embedding_df.index.names = ["current_id", "versioned_id", "x", "y", "z"]

        # embedding_df["x_nm"] = embedding_df.index.get_level_values("x") * 32
        # embedding_df["y_nm"] = embedding_df.index.get_level_values("y") * 32
        # embedding_df["z_nm"] = embedding_df.index.get_level_values("z") * 40
        # embedding_df = embedding_df.droplevel(["x", "y", "z"])
        # embedding_df.rename(
        #     columns={"x_nm": "x", "y_nm": "y", "z_nm": "z"}, inplace=True
        # )
        # mystery_offset = np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
        # embedding_df["x"] += mystery_offset[0]
        # embedding_df["y"] += mystery_offset[1]
        # embedding_df["z"] += mystery_offset[2]
        # embedding_df["x"] = embedding_df["x"].astype(int)
        # embedding_df["y"] = embedding_df["y"].astype(int)
        # embedding_df["z"] = embedding_df["z"].astype(int)

        # embedding_df.set_index(["x", "y", "z"], append=True, inplace=True)

        self.features_ = embedding_df

    # def map_to_level2(self):
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
    #                 level2_ids, attributes=["rep_coord_nm"]
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
