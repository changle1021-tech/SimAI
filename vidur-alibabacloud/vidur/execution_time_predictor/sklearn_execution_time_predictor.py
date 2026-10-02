import hashlib
import json
import os
import pickle
from abc import abstractmethod
from itertools import product
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from fasteners import InterProcessReaderWriterLock
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
    SimulationConfig,
)
from vidur.entities import Batch
from vidur.execution_time_predictor.base_execution_time_predictor import (
    BaseExecutionTimePredictor,
)
from vidur.utils.runtime_layout import configured_rank_nodes, pipeline_layer_bounds
from vidur.logger import init_logger

logger = init_logger(__name__)


class SklearnExecutionTimePredictor(BaseExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
            simulation_config=simulation_config,
        )
        os.makedirs(self._cache_dir, exist_ok=True)

        # These overheads are only for GQA models
        self._attention_prefill_batching_overhead_fraction = (
            (self._config.attention_prefill_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        self._attention_decode_batching_overhead_fraction = (
            (self._config.attention_decode_batching_overhead_fraction)
            if self._model_config.num_q_heads > self._model_config.num_kv_heads
            else 0
        )
        if self._replica_scheduler_provider == "orca":
            self._max_tokens = (
                self._config.prediction_max_tokens_per_request
                * self._config.prediction_max_batch_size
            )
        else:
            self._max_tokens = self._config.prediction_max_tokens_per_request

        num_workers = (
            self._replica_config.num_pipeline_stages
            * self._replica_config.tensor_parallel_size
        )
        devices_per_node = self._replica_config.node_config.num_devices_per_node
        self._rank_node_map = configured_rank_nodes(
            num_workers, devices_per_node, getattr(self._replica_config, "rank_node_map", None))
        self._is_multi_node = len(set(self._rank_node_map)) > 1

        (
            self._compute_input_file,       # mlp
            self._attention_input_file,
            self._all_reduce_input_file,
            self._send_recv_input_file,
            self._cpu_overhead_input_file,
        ) = self._get_input_files()
        self._model_training_hashes = {}
        self._native_tp_collectives = {}
        self._decoder_graph_profile = None
        if self._config.decoder_graph_input_file is not None:
            if self._config.backend != "vidur" or self._config.decode_attention_execution_mode != "cuda_graph":
                raise ValueError("Native decoder CUDA graph profiles require the Vidur CUDA graph runtime")
            from vidur.utils.decoder_graph_profile import DecoderGraphProfile
            capacity = self._config.cuda_graph_max_seq_len or min(
                self._config.prediction_max_tokens_per_request, self._model_config.max_position_embeddings)
            self._decoder_graph_profile = DecoderGraphProfile(
                self._config.decoder_graph_input_file, self._model_config.get_name(),
                dict(n_embd=self._model_config.embedding_dim, n_q_head=self._model_config.num_q_heads,
                     n_kv_head=self._model_config.num_kv_heads, n_expanded_embd=self._model_config.mlp_hidden_dim),
                self._replica_config.tensor_parallel_size, capacity, self._block_size,
                device=self._replica_config.device)

        # print(f"> Debug: start self._models = self._train_models() \
            # and self._predictions = self._predict_from_models()")
        
        
        # > 
        # if self._replica_config.model_name == 'hf_configs/deepseek_v3_config' and self._config.backend == 'aicb':
        if self._config.backend == 'aicb':
            self._predictions = self._predict_from_models_by_aicb()
            
            pass
        else:
            self._models = self._train_models()
            self._predictions = self._predict_from_models()
            
        # self._models = self._train_models()
        # self._predictions = self._predict_from_models()
        
        # print(f"> Debug: self._models={self._models}")
        # print(f"> Debug: self._predictions={self._predictions}")
        
        # # Check _models structure
        # print("> Debug: === _models structure ===")
        # for model_name, model in self._models.items():
        #     print(f"> Debug: Model {model_name}: {type(model)}")

        # # Check _predictions structure
        # print("\n > Debug: === _predictions structure ===")
        # for pred_name, pred_dict in self._predictions.items():
        #     print(f"> Debug: Prediction {pred_name}: Contains {len(pred_dict)} entries")
        #     # Show first few key-value pairs as examples
        #     sample_items = list(pred_dict.items())[:3]
        #     for key, value in sample_items:
        #         print(f"> Debug: Key: {key}, Value: {value}")
        #     print(f"> Debug: Key type: {type(list(pred_dict.keys())[0])}")
        #     print(f"> Debug: Value type: {type(list(pred_dict.values())[0])}")
        #     print()

    def _get_input_files(self) -> Tuple[str, str, str, str, str]:
        input_files = [
            self._config.compute_input_file,
            self._config.attention_input_file,
            self._config.all_reduce_input_file,
            self._config.send_recv_input_file,
            self._config.cpu_overhead_input_file,
        ]
        for i in range(len(input_files)):
            input_files[i] = (
                input_files[i]
                .replace("{DEVICE}", self._replica_config.device)
                .replace("{MODEL}", self._model_config.get_name())
                .replace("{NETWORK_DEVICE}", self._replica_config.network_device)
            )

        return tuple(input_files)

    def _load_compute_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        df = df.drop_duplicates()

        logger.debug(f"Length of complete compute df: {len(df)} {file_path}")
        logger.debug(f"self._num_q_heads: {self._model_config.num_q_heads}")
        logger.debug(f"self._embedding_dim: {self._model_config.embedding_dim}")
        logger.debug(f"self._mlp_hidden_dim: {self._model_config.mlp_hidden_dim}")
        logger.debug(f"self._use_gated_mlp: {self._model_config.use_gated_mlp}")
        logger.debug(f"self._vocab_size: {self._model_config.vocab_size}")
        logger.debug(
            f"self._num_tensor_parallel_workers: {self._replica_config.tensor_parallel_size}"
        )

        df = df[
            (df["n_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._model_config.use_gated_mlp)
            & (df["vocab_size"] == self._model_config.vocab_size)
            & (
                df["num_tensor_parallel_workers"]
                == self._replica_config.tensor_parallel_size
            )
        ]

        for column in [
            "time_stats.post_attention_layernorm.median",
            "time_stats.add.median",
            "time_stats.input_layernorm.median",
        ]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)
        return df

    # > Debug: added changes: ensure bug location
    def _load_attention_df(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()

        for column in [
            "time_stats.attn_kv_cache_save.median",
        ]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        # 保存筛选前的DataFrame长度
        # Save the original DataFrame length before filtering
        original_length = len(df)
        
        # 分别检查每个筛选条件
        # Check each filter condition separately
        embd_mask = (df["n_embd"] == self._model_config.embedding_dim)
        q_head_mask = (df["n_q_head"] == self._model_config.num_q_heads)
        kv_head_mask = (df["n_kv_head"] == self._model_config.num_kv_heads)
        block_size_mask = (df["block_size"] == self._block_size)
        tp_workers_mask = (
            df["num_tensor_parallel_workers"]
            == self._replica_config.tensor_parallel_size
        )
        
        # Add assertion checks to ensure each condition is met
        if original_length > 0:  ## Only check when original data is not empty
            assert embd_mask.any(), (
                f"No data found matching embedding dimension: {self._model_config.embedding_dim}. "
                f"Available values: {sorted(df['n_embd'].unique())}"
            )
            
            assert q_head_mask.any(), (
                f"No data found matching query head count: {self._model_config.num_q_heads}. "
                f"Available values: {sorted(df['n_q_head'].unique())}"
            )
            
            assert kv_head_mask.any(), (
                f"No data found matching KV head count: {self._model_config.num_kv_heads}. "
                f"Available values: {sorted(df['n_kv_head'].unique())}"
            )
            
            assert block_size_mask.any(), (
                f"No data found matching block size: {self._block_size}. "
                f"Available values: {sorted(df['block_size'].unique())}"
            )
            
            assert tp_workers_mask.any(), (
                f"No data found matching tensor parallel workers: {self._replica_config.tensor_parallel_size}. "
                f"Available values: {sorted(df['num_tensor_parallel_workers'].unique())}"
            )
        
        # Apply all filter conditions
        filtered_df = df[
            embd_mask & q_head_mask & kv_head_mask & block_size_mask & tp_workers_mask
        ]
        
        # Add logging output
        logger.info(
            f"Attention data filtering: original {original_length} rows, "
            f"filtered {len(filtered_df)} rows. "
            f"Model config: n_embd={self._model_config.embedding_dim}, "
            f"n_q_head={self._model_config.num_q_heads}, "
            f"n_kv_head={self._model_config.num_kv_heads}, "
            f"block_size={self._block_size}, "
            f"tp_workers={self._replica_config.tensor_parallel_size}"
        )
        
        self._attention_layout_aware = "decode_block_table_capacity" in filtered_df
        if self._attention_layout_aware:
            mode = self._config.decode_attention_execution_mode
            if mode not in {"cuda_graph", "eager"}:
                raise ValueError("Unknown decode attention execution mode")
            capacity = self._config.cuda_graph_max_seq_len
            if capacity is None:
                capacity = min(self._config.prediction_max_tokens_per_request,
                               self._model_config.max_position_embeddings)
            self._attention_graph_capacity = capacity
            required = {"decode_layout", "physical_batch_size"}
            if not required <= set(filtered_df.columns):
                raise ValueError("Attention layout profile is incomplete")
            is_prefill = filtered_df["is_prefill"]
            matched = filtered_df["decode_layout"].eq(mode)
            if mode == "cuda_graph":
                matched &= filtered_df["decode_block_table_capacity"].eq(capacity)
            filtered_df = filtered_df[is_prefill | matched].copy()
            if filtered_df[~filtered_df["is_prefill"]].empty:
                raise ValueError(f"No attention profile for decode layout={mode}, capacity={capacity}")
            if mode == "cuda_graph":
                decode = filtered_df[~filtered_df["is_prefill"]]
                expected = decode["batch_size"].map(lambda b: b if b <= 2 else 4 if b <= 4 else ((b + 7) // 8) * 8)
                if not decode["physical_batch_size"].eq(expected).all():
                    raise ValueError("Attention profile graph padding differs from the runtime layout")
        return filtered_df

    def _load_all_reduce_df(self, file_path: str, pipeline_stage: int = 0) -> pd.DataFrame:
        tp = self._replica_config.tensor_parallel_size
        nodes = self._rank_node_map[pipeline_stage * tp:(pipeline_stage + 1) * tp]
        aliases = {}
        placement = [aliases.setdefault(n, len(aliases)) for n in nodes]
        df = self._read_input_file(file_path)
        selected = df[(df["num_workers"] == tp) & df["collective"].eq("all_reduce")].copy()
        if "rank_node_map" in selected:
            encoded = json.dumps(placement, separators=(",", ":"))
            selected = selected[selected["rank_node_map"].map(
                lambda x: json.dumps(json.loads(x), separators=(",", ":"))) == encoded]
        else:
            counts = [nodes.count(n) for n in set(nodes)]
            if len(set(counts)) != 1:
                raise ValueError("Uneven cross-node TP groups require an explicit rank_node_map in collective profiles")
            selected = selected[selected["devices_per_node"] == counts[0]]
        if len(set(nodes)) > 1:
            transport = getattr(self._config, "network_transport", None)
            if not transport or "network_transport" not in selected or "rank_node_map" not in selected:
                raise ValueError("Cross-node TP profiles require explicit rank_node_map and verified network_transport; legacy node counts do not identify the network")
            selected = selected[selected["network_transport"].eq(transport)]
        if selected.empty:
            raise ValueError(f"No all-reduce profile for TP={tp}, stage={pipeline_stage}, group placement={placement}")
        if "profile_kind" in selected:
            native=selected["profile_kind"].eq("native_collective_v1")
            if native.any() and not native.all():
                raise ValueError("Native collective profiles cannot be mixed with legacy timing semantics")
        return selected

    def _load_send_recv_df(self, file_path: str) -> pd.DataFrame:
        if self._is_multi_node:
            devices_per_node = 1
        else:
            devices_per_node = 2

        df = self._read_input_file(file_path)
        filtered_df = df[
            (df["collective"] == "send_recv")
            & (df["devices_per_node"] == devices_per_node)
        ]
        return filtered_df

    def _load_cpu_overhead_df(self, file_path: str) -> pd.DataFrame:
        df = self._read_input_file(file_path)
        pp = self._replica_config.num_pipeline_stages
        if "pipeline_parallel_degree" not in df:
            if pp != 1:
                raise ValueError(
                    f"PP={pp} requires a CPU profile with pipeline_parallel_degree; "
                    "legacy CPU tables were collected at PP=1"
                )
            df["pipeline_parallel_degree"] = 1
        degrees = pd.to_numeric(df["pipeline_parallel_degree"], errors="raise")
        filtered = df[
            (df["model_name"] == self._model_config.get_name())
            & (df["tensor_parallel_degree"] == self._replica_config.tensor_parallel_size)
            & (degrees == pp)
        ].copy()
        expected_nodes = getattr(self, "_rank_node_map", [0] * (self._replica_config.tensor_parallel_size * pp))
        if "rank_node_map" in filtered:
            encoded = json.dumps(expected_nodes, separators=(",", ":"))
            legacy = json.dumps([0] * len(expected_nodes), separators=(",", ":"))
            maps = filtered["rank_node_map"].fillna(legacy).map(
                lambda value: json.dumps(json.loads(value), separators=(",", ":")))
            filtered = filtered[maps == encoded].copy()
        elif len(set(expected_nodes)) > 1:
            raise ValueError("Cross-node CPU profiling must include actual rank_node_map; single-node data cannot be reused")
        if len(set(expected_nodes)) > 1:
            transport = getattr(getattr(self, "_config", None), "network_transport", None)
            if not transport or "network_transport" not in filtered:
                raise ValueError("Cross-node CPU profiles require verified network_transport; socket and IB measurements cannot be mixed")
            filtered = filtered[filtered["network_transport"].eq(transport)].copy()
        if "profile_loop_mode" in filtered:
            desired_loop = self._config.execution_loop_mode
            filtered = filtered[filtered["profile_loop_mode"].fillna("direct").eq(desired_loop)].copy()
        elif getattr(getattr(self, "_config", None), "execution_loop_mode", "direct") != "direct":
            raise ValueError("Serving-loop CPU simulation requires a serving-loop profile; inner-step data omits background scheduling")
        if "enforce_eager" in filtered:
            eager = self._config.decode_attention_execution_mode == "eager"
            modes = filtered["enforce_eager"].map(lambda x: str(x).lower() == "true")
            filtered = filtered[modes == eager].copy()
        if "profile_schema_version" in filtered:
            versions = pd.to_numeric(filtered["profile_schema_version"], errors="coerce").fillna(1)
            filtered = filtered[versions == versions.max()].copy()
        if filtered.empty:
            raise ValueError(f"No CPU profile for model={self._model_config.get_name()}, "
                             f"TP={self._replica_config.tensor_parallel_size}, PP={pp}, rank_node_map={expected_nodes}")
        phase_profile = (
            "prefill_tokens_per_request" in filtered
            and filtered["prefill_tokens_per_request"].notna().all()
        )
        self._cpu_profile_feature_cols = ["batch_size"]
        self._cpu_profile_prefill_sizes = [0]
        self._has_pp_handoff_profile = False
        self._pp_handoff_model_names = []
        self._graph_staging_model_names = []
        stage_names = [f"graph_input_staging_stage_{i}" for i in range(pp)]
        if all(name + "_mean" in filtered and filtered[name + "_mean"].notna().all() for name in stage_names):
            self._graph_staging_model_names = stage_names
        self._has_engine_bookkeeping_profile = (
            "engine_bookkeeping_mean" in filtered
            and filtered["engine_bookkeeping_mean"].notna().all())
        if phase_profile:
            self._cpu_profile_feature_cols.append("prefill_tokens_per_request")
            self._cpu_profile_prefill_sizes = sorted(
                filtered["prefill_tokens_per_request"].unique().tolist())
            if "phase" not in filtered or set(filtered["phase"]) != {"prefill", "decode"}:
                raise ValueError("CPU phase profile must include both prefill and decode")
            if pp > 1:
                if ("pp_handoff_e2e_mean" not in filtered
                        or filtered["pp_handoff_e2e_mean"].isna().any()):
                    raise ValueError("PP phase profile is missing complete handoff timings")
                self._has_pp_handoff_profile = True
                names = [f"pp_handoff_boundary_{i}" for i in range(pp - 1)]
                if all(name + "_mean" in filtered and filtered[name + "_mean"].notna().all() for name in names):
                    self._pp_handoff_model_names = names
                elif pp > 2:
                    raise ValueError("PP>2 requires per-boundary profiles; aggregate handoff cannot describe mixed local and cross-node links")
        elif pp > 1:
            raise ValueError("PP CPU profiles require separate prefill/decode timings")
        self._cpu_observed_points = {}
        self._pipeline_protocol_model_names = []
        self._eager_dispatch_programs = []
        if ("profile_schema_version" in filtered and filtered["profile_schema_version"].min() >= 8
                and len(set(expected_nodes)) == 1):
            names = ["pipeline_result_return", "executor_pre_dispatch"] + [
                f"pipeline_{kind}_stage_{i}" for kind in ["input_work", "input_dispatch", "post_receive"]
                for i in range(pp)]
            if any(name + "_mean" not in filtered or filtered[name + "_mean"].isna().any() for name in names):
                raise ValueError("Incomplete named pipeline protocol profile")
            self._pipeline_protocol_model_names = names
            prefill = filtered[filtered["phase"].eq("prefill")]
            for stage in range(pp):
                column = f"eager_dispatch_program_stage_{stage}"
                if column not in prefill or prefill[column].isna().any():
                    raise ValueError("Missing eager CPU launch program")
                points = {}
                for _, row in prefill.iterrows():
                    key = (int(row["batch_size"]), float(row["prefill_tokens_per_request"]))
                    program = json.loads(row[column])
                    if key in points:
                        raise ValueError("Duplicate eager CPU program point; aggregate repetitions during collection")
                    points[key] = program
                self._eager_dispatch_programs.append(points)
        if phase_profile and "profile_schema_version" in filtered and filtered["profile_schema_version"].min() >= 5:
            for column in filtered:
                if column.endswith("_mean") and filtered[column].notna().all():
                    points = filtered.groupby(self._cpu_profile_feature_cols)[column].mean()
                    self._cpu_observed_points[column[:-5]] = {
                        key if isinstance(key, tuple) else (key,): float(value)
                        for key, value in points.items()}
        return filtered

    def _read_input_file(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()
        return df

    def _get_compute_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    def _get_attention_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = df_with_derived_features[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)
        df_with_derived_features["is_decode"] = (
            df_with_derived_features["prefill_chunk_size"] == 0
        )
        df_with_derived_features["prefill_chunk_size_squared"] = (
            df_with_derived_features["prefill_chunk_size"] ** 2
        )
        return df_with_derived_features

    def _get_all_reduce_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        # convert bytes to num tokens
        # each token is of size 2 * h bytes
        
        # 单个allreduce原语
        # Single allreduce primitive
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_send_recv_df_with_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        df_with_derived_features["num_tokens"] = (
            df_with_derived_features["size"] / self._model_config.embedding_dim / 2
        )
        return df_with_derived_features

    def _get_cpu_overhead_df_with_derived_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        df_with_derived_features = df.copy()
        return df_with_derived_features

    @staticmethod
    def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Handling the case where y_true is 0 separately to avoid division by zero
        zero_true_mask = y_true == 0
        non_zero_true_mask = ~zero_true_mask

        # For non-zero true values, calculate the absolute percentage error
        error = np.zeros_like(y_true, dtype=float)  # using float instead of np.float
        error[non_zero_true_mask] = (
            np.abs(
                (y_true[non_zero_true_mask] - y_pred[non_zero_true_mask])
                / y_true[non_zero_true_mask]
            )
            * 100
        )

        # For zero true values, if prediction is also 0, error is 0, else it is 100
        error[zero_true_mask] = np.where(y_pred[zero_true_mask] == 0, 0, 100)

        # Return the mean of the absolute percentage errors
        return np.mean(error)

    def _get_scorer(self) -> Any:
        return make_scorer(
            SklearnExecutionTimePredictor.mean_absolute_percentage_error,
            greater_is_better=False,
        )

    @abstractmethod
    def _get_grid_search_params(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_estimator(self) -> BaseEstimator:
        pass

    def _get_model_hash(
        self, model_name: str, df: pd.DataFrame = None, target_col: str = None
    ) -> str:
        config = dict(self.to_dict())
        timing_name = model_name.split("_phase_", 1)[0]
        cpu_name = (timing_name in {"schedule", "sampler_e2e", "prepare_inputs_e2e",
                    "process_model_outputs", "ray_comm_time", "pp_handoff_e2e", "engine_bookkeeping"}
                    or timing_name.startswith(("pp_handoff_boundary_", "graph_input_staging_stage_", "pipeline_", "executor_pre_dispatch")))
        if not cpu_name:
            config.pop("cpu_overhead_input_file", None)
        if not cpu_name and timing_name not in {"all_reduce", "send_recv"} and not timing_name.startswith("all_reduce_stage_"):
            config.pop("network_transport", None)
        if timing_name not in {"all_reduce", "send_recv"} and not timing_name.startswith("all_reduce_stage_"):
            config.pop("all_reduce_input_file", None)
            config.pop("send_recv_input_file", None)
        config_str = str(config)
        if timing_name in {"schedule", "sampler_e2e", "prepare_inputs_e2e",
                          "process_model_outputs", "ray_comm_time", "pp_handoff_e2e", "engine_bookkeeping"} or timing_name.startswith(("pp_handoff_boundary_", "graph_input_staging_stage_", "pipeline_", "executor_pre_dispatch")):
            config_str += f"_cpu_schema=5_pp={self._replica_config.num_pipeline_stages}_placement={getattr(self, '_rank_node_map', None)}"

        if model_name in {"attn_decode", "attn_prefill", "attn_kv_cache_save"} and getattr(self, "_attention_layout_aware", False):
            config_str += f"_attention_layout={self._config.decode_attention_execution_mode}_capacity={self._attention_graph_capacity}"

        if model_name == "all_reduce" or model_name.startswith("all_reduce_stage_"):
            config_str += f"_tp_group_placement={self._rank_node_map}"

        if df is None:
            combined_str = f"{config_str}_{model_name}"
        else:
            df_hash_str = hashlib.md5(df.to_json().encode("utf-8")).hexdigest()
            combined_str = f"{config_str}_{model_name}_{df_hash_str}"

        if target_col is not None:
            combined_str += f"_target={target_col}"
        return hashlib.md5(combined_str.encode("utf-8")).hexdigest()[0:8]

    def _load_model_from_cache(self, model_name: str, model_hash: str) -> BaseEstimator:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).read_lock():
            if self._config.no_cache:
                return
            # check if model is in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            if not os.path.exists(cache_file):
                return

            logger.debug(f"Found model {model_name} in cache")
            model = pickle.load(open(cache_file, "rb"))
            return model

    def _store_model_in_cache(
        self, model_name: str, model_hash: str, model: BaseEstimator
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_model_lock.file"
        ).write_lock():
            # store model in cache
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}.pkl"
            pickle.dump(model, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def _store_training_prediction_data(
        self,
        model_name: str,
        model_hash: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        model: BaseEstimator,
    ) -> None:
        df = df.copy()

        # convert the df to list of tuples
        df["prediction"] = model.predict(df[feature_cols])

        # store the prediction data
        df[feature_cols + [target_col, "prediction"]].to_csv(
            f"{self._cache_dir}/{model_name}_{model_hash}_training_predictions.csv",
            index=False,
        )

    def _train_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
    ) -> BaseEstimator:
        if len(df) == 0:
            raise Exception(f"Training data for model {model_name} is empty")

        model_hash = self._get_model_hash(model_name, df, target_col=target_col)
        self._model_training_hashes[model_name] = model_hash

        cached_model = self._load_model_from_cache(model_name, model_hash)
        if cached_model:
            return cached_model

        model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        if len(df) < self._config.k_fold_cv_splits:
            cv = 2
        else:
            cv = self._config.k_fold_cv_splits

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )

        # we don't create a train/test split, because we want to use all data for training
        # and we don't care about overfitting, because we only want to predict execution time within the same domain
        X, y = df[feature_cols], df[target_col]

        grid_search.fit(X, y)
        score = grid_search.score(X, y)

        logger.info(
            f"Trained model {model_name} and found best parameters: {grid_search.best_params_} "
            f"with mean absolute percentage error (MEAP) {-score}%"
        )

        self._store_model_in_cache(model_name, model_hash, grid_search.best_estimator_)

        self._store_training_prediction_data(
            model_name=model_name,
            model_hash=model_hash,
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            model=grid_search.best_estimator_,
        )
        return grid_search.best_estimator_

    def _store_model_predication_cache(
        self, model_name: str, model_hash: str, predictions: Dict[Tuple, float]
    ) -> None:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_prediction_lock.file"
        ).write_lock():
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}_predictions.pkl"
            pickle.dump(
                predictions, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
            )

    def _load_model_predication_cache(
        self, model_name: str, model_hash: str
    ) -> Dict[Tuple, float]:
        with InterProcessReaderWriterLock(
            f"{self._cache_dir}/{model_hash}_prediction_lock.file"
        ).read_lock():
            if self._config.no_cache:
                return
            cache_file = f"{self._cache_dir}/{model_name}_{model_hash}_predictions.pkl"

            if not os.path.exists(cache_file):
                return

            logger.debug(f"Found model {model_name} predictions in cache")

            predictions = pickle.load(open(cache_file, "rb"))
            return predictions

    def _get_model_prediction(
        self, model_name: str, model: BaseEstimator, X: pd.DataFrame
    ) -> Dict[Tuple, float]:
        X = X.copy()

        training_hash = self._model_training_hashes.get(model_name)
        if training_hash is None:
            training_hash = hashlib.md5(
                pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
            ).hexdigest()
        prediction_key = f"{training_hash}_{X.to_json()}"
        model_hash = hashlib.md5(prediction_key.encode("utf-8")).hexdigest()[0:8]

        cached_predictions = self._load_model_predication_cache(model_name, model_hash)
        if cached_predictions:
            return cached_predictions

        logger.info(f"Predicting execution time for model {model_name}")

        predictions_array = model.predict(X)

        # turn this into a dict, so we can store use it as a cache
        # the key is tuple for each row of X
        predictions = dict(zip([tuple(x) for x in X.values], predictions_array))

        self._store_model_predication_cache(model_name, model_hash, predictions)

        X["prediction"] = predictions_array
        X.to_csv(
            f"{self._cache_dir}/{model_name}_{model_hash}_predictions.csv",
            index=False,
        )

        return predictions

    def _train_compute_models(self) -> Dict[str, BaseEstimator]:
        # print(f"> Debug: self._compute_input_file={self._compute_input_file}")
        compute_df = self._load_compute_df(self._compute_input_file)
        # print(f"> Debug: self._compute_input_file={self._compute_input_file}, compute_df={compute_df}")
        compute_df = self._get_compute_df_with_derived_features(compute_df)
        # print(f"> Debug: self._compute_input_file={self._compute_input_file}, compute_df={compute_df}")

        models = {}
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_rope",
            "add",
            "emb",
        ]

        for boundary in ["first_layernorm", "final_layernorm"]:
            if f"time_stats.{boundary}.median" in compute_df:
                model_names.append(boundary)

        if self._model_config.rope_theta is None:
            model_names.remove("attn_rope")

        for model_name in model_names:
            logger.debug(
                f"Training model {model_name}, size of training data: {len(compute_df)}"
            )
            models[model_name] = self._train_model(
                model_name=model_name,
                df=compute_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        attention_df = self._load_attention_df(self._attention_input_file)
        # print(f"> Debug: self._attention_input_file={self._attention_input_file}, attention_df={attention_df}")
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        

        model_names = [
            "attn_kv_cache_save",
        ]

        for model_name in model_names:
            models[model_name] = self._train_model(
                model_name=model_name,
                df=attention_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        if self._replica_config.num_pipeline_stages > 1 and not getattr(self, "_has_pp_handoff_profile", False):
            send_recv_df = self._load_send_recv_df(self._send_recv_input_file)
            send_recv_df = self._get_send_recv_df_with_derived_features(send_recv_df)

            models["send_recv"] = self._train_model(
                model_name="send_recv",
                df=send_recv_df,
                feature_cols=["num_tokens"],
                target_col="time_stats.send_recv.mean",
            )

        if self._replica_config.tensor_parallel_size > 1:
            tp = self._replica_config.tensor_parallel_size
            patterns = []
            for stage in range(self._replica_config.num_pipeline_stages):
                aliases = {}
                group = self._rank_node_map[stage * tp:(stage + 1) * tp]
                patterns.append(tuple(aliases.setdefault(n, len(aliases)) for n in group))
            uniform = len(set(patterns)) == 1
            self._tp_group_model_names = ["all_reduce"] * len(patterns) if uniform else [
                f"all_reduce_stage_{i}" for i in range(len(patterns))]
            for stage, name in enumerate(self._tp_group_model_names):
                if name in models:
                    continue
                all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file, pipeline_stage=stage)
                if "profile_kind" in all_reduce_df and all_reduce_df["profile_kind"].eq("native_collective_v1").all():
                    from vidur.utils.native_collective_profile import NativeCollectiveProfile
                    self._native_tp_collectives[stage] = NativeCollectiveProfile(
                        all_reduce_df.to_dict("records"), self._model_config.embedding_dim,
                        device=self._replica_config.device)
                    continue
                all_reduce_df = self._get_all_reduce_df_with_derived_features(all_reduce_df)
                models[name] = self._train_model(
                    model_name=name, df=all_reduce_df, feature_cols=["num_tokens"],
                    target_col="time_stats.all_reduce.mean")

        return models

    def _train_cpu_overhead_models(self) -> Dict[str, BaseEstimator]:
        if self._config.skip_cpu_overhead_modeling:
            return {}
        df = self._get_cpu_overhead_df_with_derived_features(
            self._load_cpu_overhead_df(self._cpu_overhead_input_file))
        names = ["schedule", "sampler_e2e", "prepare_inputs_e2e",
                 "process_model_outputs", "ray_comm_time"]
        if self._has_pp_handoff_profile:
            names.extend(self._pp_handoff_model_names or ["pp_handoff_e2e"])
        if self._has_engine_bookkeeping_profile:
            names.append("engine_bookkeeping")
        names.extend(self._graph_staging_model_names)
        names.extend(getattr(self, "_pipeline_protocol_model_names", []))
        enhanced = len(self._cpu_profile_feature_cols) > 1
        self._cpu_phase_models = enhanced
        models = {}
        for name in names:
            for phase in (["prefill", "decode"] if enhanced else [None]):
                part = df[df["phase"] == phase].copy() if phase else df
                model_name = f"{name}_phase_{phase}" if phase else name
                if phase and len(part) < 2:
                    raise ValueError("Each CPU phase needs at least two profiling points for regression")
                models[model_name] = self._train_model(
                    model_name=model_name, df=part, feature_cols=self._cpu_profile_feature_cols,
                    target_col=name + ("_mean" if enhanced or name == "ray_comm_time" else "_median"))
        return models

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        attention_df = self._load_attention_df(self._attention_input_file)
        attention_df = self._get_attention_df_with_derived_features(attention_df)
        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]

        models = {}

        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"]
            + chunked_prefill_df["prefill_chunk_size"]
        )

        models["attn_prefill"] = self._train_model(
            model_name="attn_prefill",
            df=prefill_df,
            feature_cols=["kv_cache_size", "prefill_chunk_size_squared"],
            target_col="time_stats.attn_prefill.median",
        )

        models["attn_decode"] = self._train_model(
            model_name="attn_decode",
            df=decode_df,
            feature_cols=["batch_size", "kv_cache_size"],
            target_col="time_stats.attn_decode.median",
        )

        return models

    def _train_models(self) -> Dict[str, BaseEstimator]:
        # Resolve CPU protocol coverage before selecting a bare transfer model.
        if not self._config.skip_cpu_overhead_modeling:
            self._load_cpu_overhead_df(self._cpu_overhead_input_file)
        models = self._train_compute_models()
        models.update(self._train_cpu_overhead_models())
        models.update(self._train_attention_layer_models())

        return models

    def _predict_for_compute_models_by_aicb(self) -> Dict[str, Any]:
        # NOTE: AICB backend uses empirical linear formulas (time = a * batch_size + b)
        # instead of ML model predictions. These formulas are derived from profiling data
        # on specific hardware (H20) and represent the designed approach for AICB backend,
        # not a temporary placeholder.
        # 注意: AICB 后端使用基于经验的线性公式 (time = a * batch_size + b)
        # 代替 ML 模型预测。这些公式来自特定硬件 (H20) 的 profiling 数据，
        # 是 AICB 后端的设计方案，而非临时占位代码。
        
        # Store prediction results / 存储预测结果
        predictions = {}

        # Define compute layer model names for prediction
        # 定义需要预测的计算层模型名
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            # "attn_rope",  # RoPE layer modeling not enabled / 当前未启用RoPE层建模
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
        ]

        # Add send/recv comm model if pipeline parallelism exists
        # 若存在流水线并行，则加入send/recv通信模型
        if self._replica_config.num_pipeline_stages > 1:
            model_names.append("send_recv")

        # Add all_reduce comm model if tensor parallelism exists
        # 若存在张量并行，则加入all_reduce通信模型
        if self._replica_config.tensor_parallel_size > 1:
            model_names.append("all_reduce")

        # Generate range from 1 to max tokens for batch prediction
        # 生成从1到最大token数的范围，用于批量预测
        num_token_range = np.arange(1, self._max_tokens + 1)
        # Construct input DataFrame / 构造输入DataFrame
        X = pd.DataFrame({"num_tokens": num_token_range})

        # Predict and cache results for each compute model
        # 对每个计算模型进行预测，并将结果缓存
        for model_name in model_names:
            # Create simulated predictions with linear growth based on num_tokens
            # Set different base time and growth rate per model type
            # 创建模拟的预测值，基于num_tokens线性增长
            # 根据模型类型设置不同的基础时间和增长率
            if model_name in ["attn_pre_proj", "attn_post_proj"]:
                # Attention projection: base 0.001ms, +0.0001ms per token
                # 注意力投影层：基础时间0.001ms，每token增加0.0001ms
                predictions[model_name] = {
                    (num_tokens,): 0.001 + 0.0001 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name in ["mlp_up_proj", "mlp_down_proj"]:
                # MLP layer: base 0.0015ms, +0.00015ms per token
                # MLP层：基础时间0.0015ms，每token增加0.00015ms
                predictions[model_name] = {
                    (num_tokens,): 0.0015 + 0.00015 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name == "mlp_act":
                # MLP activation: base 0.0005ms, +0.00005ms per token
                # MLP激活层：基础时间0.0005ms，每token增加0.00005ms
                predictions[model_name] = {
                    (num_tokens,): 0.0005 + 0.00005 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name == "attn_kv_cache_save":
                # KV cache save: base 0.0002ms, +0.00002ms per token
                # KV缓存保存：基础时间0.0002ms，每token增加0.00002ms
                predictions[model_name] = {
                    (num_tokens,): 0.0002 + 0.00002 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name in ["input_layernorm", "post_attention_layernorm"]:
                # LayerNorm: base 0.0003ms, +0.00003ms per token
                # LayerNorm层：基础时间0.0003ms，每token增加0.00003ms
                predictions[model_name] = {
                    (num_tokens,): 0.0003 + 0.00003 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name == "add":
                # Add op: base 0.0001ms, +0.00001ms per token
                # Add操作：基础时间0.0001ms，每token增加0.00001ms
                predictions[model_name] = {
                    (num_tokens,): 0.0001 + 0.00001 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name == "send_recv":
                # Send/Recv comm: base 0.01ms, +0.001ms per token
                # Send/Recv通信：基础时间0.01ms，每token增加0.001ms
                predictions[model_name] = {
                    (num_tokens,): 0.01 + 0.001 * num_tokens 
                    for num_tokens in num_token_range
                }
            elif model_name == "all_reduce":
                # All-reduce comm: base 0.02ms, +0.002ms per token
                # All-reduce通信：基础时间0.02ms，每token增加0.002ms
                predictions[model_name] = {
                    (num_tokens,): 0.02 + 0.002 * num_tokens 
                    for num_tokens in num_token_range
                }

        # Predict and cache results for each compute model
        # 对每个计算模型进行预测，并将结果缓存
        # for model_name in model_names:
        #     model = self._models[model_name]
        #     predictions[model_name] = self._get_model_prediction(model_name, model, X)

        # Return all compute layer predictions (for later lookup)
        # 返回所有计算相关层的预测结果（用于后续查表）
        return predictions

    def _predict_for_compute_models(self) -> Dict[str, Any]:
        predictions = {}

        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "attn_rope",
            "attn_kv_cache_save",
            "input_layernorm",
            "post_attention_layernorm",
            "add",
            "emb",
        ]

        for boundary in ["first_layernorm", "final_layernorm"]:
            if boundary in self._models:
                model_names.append(boundary)

        if self._model_config.rope_theta is None:
            model_names.remove("attn_rope")

        if "send_recv" in self._models:
            model_names.append("send_recv")

        if self._replica_config.tensor_parallel_size > 1:
            model_names.extend(n for n in self._tp_group_model_names if n in self._models)

        num_token_range = np.arange(1, self._max_tokens + 1)
        X = pd.DataFrame({"num_tokens": num_token_range})

        # 将[1, max_tokens]范围内的batched_token_size用所有的model预测一遍，并缓存结果
        # Predict batched_token_size in the range [1, max_tokens] with all models and cache the results
        for model_name in model_names:
            model = self._models[model_name]
            predictions[model_name] = self._get_model_prediction(model_name, model, X)

        # predictions用于在实际预测时查表
        # predictions are used for lookup during actual prediction
        return predictions


    def _predict_for_cpu_overhead_models_by_aicb(self) -> Dict[str, Any]:
        # Skip if CPU overhead modeling is disabled / 若跳过CPU开销建模，则直接返回空
        if self._config.skip_cpu_overhead_modeling:
            return {}

        # Store CPU overhead predictions / 存储CPU开销预测结果
        predictions = {}

        # CPU-related overhead model names / CPU相关的开销模型名称
        model_names = [
            "schedule",
            "sampler_e2e",
            "prepare_inputs_e2e",
            "process_model_outputs",
            "ray_comm_time",
        ]

        # Batch size range: 1 to max prediction batch size
        # 批处理大小范围：1 到 最大预测批大小
        batch_size_range = np.arange(1, self._config.prediction_max_batch_size + 1)
        # Construct input data / 构造输入数据
        X = pd.DataFrame({"batch_size": batch_size_range})

        # Predict for each CPU overhead model / 对每个CPU开销模型进行预测
        # for model_name in model_names:
        #     model = self._models[model_name]
        #     predictions[model_name] = self._get_model_prediction(model_name, model, X)


        # Predict for each CPU overhead model / 对每个CPU开销模型进行预测
        for model_name in model_names:
            # Generate simulated values for each model / 为每个模型生成模拟值
            if model_name == "schedule":
                # Scheduling overhead: base 0.5ms, +0.05ms per batch
                # 调度开销：基础时间0.5ms，每增加一个batch增加0.05ms
                predictions[model_name] = {
                    (batch_size,): 0.5 + 0.05 * batch_size
                    for batch_size in batch_size_range
                }
            elif model_name == "sampler_e2e":
                # Sampler e2e overhead: base 1.0ms, +0.1ms per batch
                # 采样端到端开销：基础时间1.0ms，每增加一个batch增加0.1ms
                predictions[model_name] = {
                    (batch_size,): 1.0 + 0.1 * batch_size
                    for batch_size in batch_size_range
                }
            elif model_name == "prepare_inputs_e2e":
                # Prepare inputs e2e overhead: base 0.8ms, +0.08ms per batch
                # 准备输入端到端开销：基础时间0.8ms，每增加一个batch增加0.08ms
                predictions[model_name] = {
                    (batch_size,): 0.8 + 0.08 * batch_size
                    for batch_size in batch_size_range
                }
            elif model_name == "process_model_outputs":
                # Process model outputs overhead: base 0.3ms, +0.03ms per batch
                # 处理模型输出开销：基础时间0.3ms，每增加一个batch增加0.03ms
                predictions[model_name] = {
                    (batch_size,): 0.3 + 0.03 * batch_size
                    for batch_size in batch_size_range
                }
            elif model_name == "ray_comm_time":
                # Ray comm time: base 0.2ms, +0.02ms per batch
                # Ray通信时间：基础时间0.2ms，每增加一个batch增加0.02ms
                predictions[model_name] = {
                    (batch_size,): 0.2 + 0.02 * batch_size
                    for batch_size in batch_size_range
                }

        # Return CPU overhead predictions / 返回CPU开销部分的预测结果
        return predictions

    def _predict_for_cpu_overhead_models(self) -> Dict[str, Any]:
        if self._config.skip_cpu_overhead_modeling:
            return {}
        names = ["schedule", "sampler_e2e", "prepare_inputs_e2e",
                 "process_model_outputs", "ray_comm_time"]
        if self._has_pp_handoff_profile:
            names.extend(self._pp_handoff_model_names or ["pp_handoff_e2e"])
        if self._has_engine_bookkeeping_profile:
            names.append("engine_bookkeeping")
        names.extend(self._graph_staging_model_names)
        names.extend(getattr(self, "_pipeline_protocol_model_names", []))
        sizes = range(1, self._config.prediction_max_batch_size + 1)
        if len(self._cpu_profile_feature_cols) == 1:
            X = pd.DataFrame({"batch_size": list(sizes)})
        else:
            X = pd.DataFrame(list(product(sizes, self._cpu_profile_prefill_sizes)),
                             columns=self._cpu_profile_feature_cols)
        predictions = {}
        for name in names:
            if self._cpu_phase_models:
                values = {}
                for phase in ["prefill", "decode"]:
                    model_name = f"{name}_phase_{phase}"
                    mask = X["prefill_tokens_per_request"] > 0
                    subset = X[mask if phase == "prefill" else ~mask]
                    values.update(self._get_model_prediction(model_name, self._models[model_name], subset))
                predictions[name] = values
            else:
                predictions[name] = self._get_model_prediction(name, self._models[name], X)
        return predictions

    def _get_cpu_profile_prediction(self, name: str, batch: Batch) -> float:
        key = (batch.size,)
        if len(self._cpu_profile_feature_cols) > 1:
            key += (batch.num_prefill_tokens / batch.size,)
        observed = getattr(self, "_cpu_observed_points", {}).get(name, {})
        if key in observed:
            return observed[key]
        values = self._predictions[name]
        if key not in values:
            # Predict unseen prompt lengths without allocating a full token grid.
            X = pd.DataFrame([key], columns=self._cpu_profile_feature_cols)
            model_name = name
            if getattr(self, "_cpu_phase_models", False):
                model_name += "_phase_prefill" if batch.num_prefill_tokens else "_phase_decode"
            values[key] = float(self._models[model_name].predict(X)[0])
        return values[key]

    def _predict_for_attention_layer_models_by_aicb(self) -> Dict[str, Any]:
        # Store attention layer predictions / 存储注意力层预测结果
        predictions = {}

        # Decode batch size enumeration range / 解码阶段的batch size枚举范围
        decode_batch_size_range = np.arange(
            1, self._config.prediction_max_batch_size + 1
        )
        # KV cache size range, incremented by block granularity
        # kv缓存大小的枚举范围，按block粒度递增
        decode_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,       # KV cache stored in blocks / kv cache存储为block，这里就是block_size
        )
        # Decode prefill_chunk_size is fixed at 0 / 解码时prefill chunk size固定为0
        decode_prefill_chunk_size_range = [0]
        # Generate all combinations (Cartesian product)
        # 生成所有组合（笛卡尔积）
        decode_batch_size, decode_kv_cache_size, decode_prefill_chunk_size = zip(
            *product(
                decode_batch_size_range,
                decode_kv_cache_size_range,
                decode_prefill_chunk_size_range,
            )
        )

        # Prefill only supports batch_size=1 (single request typically)
        # Prefill阶段仅支持batch_size=1（通常单请求）
        prefill_batch_size_range = [1]
        # Also enumerate KV cache size by block granularity
        # 同样按block粒度枚举kv缓存大小
        prefill_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        # Prefill chunk size from 1 to max / Prefill chunk大小从1到最大允许值
        prefill_prefill_chunk_size_range = np.arange(
            1, self._config.prediction_max_prefill_chunk_size + 1
        )
        # Generate all prefill parameter combinations
        # 生成所有prefill参数组合
        prefill_batch_size, prefill_kv_cache_size, prefill_prefill_chunk_size = zip(
            *product(
                prefill_batch_size_range,
                prefill_kv_cache_size_range,
                prefill_prefill_chunk_size_range,
            )
        )

        # Merge decode and prefill combinations into one DataFrame
        # 合并decode和prefill的所有参数组合成一个DataFrame
        attention_df = pd.DataFrame(
            {
                "batch_size": decode_batch_size + prefill_batch_size,
                "kv_cache_size": decode_kv_cache_size + prefill_kv_cache_size,
                "prefill_chunk_size": decode_prefill_chunk_size
                + prefill_prefill_chunk_size,
            }
        )

        # Mark decode stage (chunk_size == 0) / 标记是否为decode阶段
        attention_df["is_decode"] = attention_df["prefill_chunk_size"] == 0
        # num_tokens = max(prefill_chunk_size, batch_size)
        # num_tokens取prefill_chunk_size和batch_size的最大值
        attention_df["num_tokens"] = attention_df[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)
        # Add squared prefill chunk size feature (for model input)
        # 添加prefill chunk大小的平方项（用于模型输入）
        attention_df["prefill_chunk_size_squared"] = (
            attention_df["prefill_chunk_size"] ** 2
        )

        # Split into prefill and decode subsets / 分离出prefill和decode的数据子集
        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]
        # Filter prefill data with existing cache / 进一步筛选有缓存的prefill数据
        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()
        # Calculate total prefill token count / 计算总的prefill token数量
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"]
            + chunked_prefill_df["prefill_chunk_size"]
        )
        

        # Predict prefill attention time using simulated values (based on kv_cache and chunk^2)
        # 使用模拟值预测prefill注意力时间（基于kv缓存和chunk^2）
        prefill_data = prefill_df[["kv_cache_size", "prefill_chunk_size_squared"]].values
        predictions["attn_prefill"] = {}
        for kv_cache_size, prefill_chunk_size_squared in prefill_data:
            # Prefill time based on kv_cache size and chunk size squared
            # base 0.01ms + kv_cache/100 * 0.001ms + chunk_size_sq/10000 * 0.0001ms
            # Prefill时间基于kv_cache大小和chunk大小平方计算
            # 基础时间0.01ms + kv_cache每增加100增加0.001ms + chunk_size_squared每增加10000增加0.0001ms
            time = 0.01 + (kv_cache_size * 0.001 / 100) + (prefill_chunk_size_squared * 0.0001 / 10000)
            predictions["attn_prefill"][(int(kv_cache_size), int(prefill_chunk_size_squared))] = time

        # Predict decode attention time using simulated values (based on batch_size and kv_cache)
        # 使用模拟值预测decode注意力时间（基于batch_size和kv缓存）
        decode_data = decode_df[["batch_size", "kv_cache_size"]].values
        predictions["attn_decode"] = {}
        for batch_size, kv_cache_size in decode_data:
            # Decode time based on batch_size and kv_cache size
            # base 0.005ms + 0.002ms per batch_size + kv_cache/100 * 0.0005ms
            # Decode时间基于batch_size和kv_cache大小计算
            # 基础时间0.005ms + batch_size每增加1增加0.002ms + kv_cache_size每增加100增加0.0005ms
            time = 0.005 + (batch_size * 0.002) + (kv_cache_size * 0.0005 / 100)
            predictions["attn_decode"][(int(batch_size), int(kv_cache_size))] = time

        # # Predict prefill attention time using trained model (based on kv_cache and chunk^2)
        # # 使用训练好的模型预测prefill注意力时间（基于kv缓存和chunk^2）
        # predictions["attn_prefill"] = self._get_model_prediction(
        #     "attn_prefill",
        #     self._models["attn_prefill"],
        #     prefill_df[["kv_cache_size", "prefill_chunk_size_squared"]],
        # )

        # # Predict decode attention time using trained model (based on batch_size and kv_cache)
        # # 使用训练好的模型预测decode注意力时间（基于batch_size和kv缓存）
        # predictions["attn_decode"] = self._get_model_prediction(
        #     "attn_decode",
        #     self._models["attn_decode"],
        #     decode_df[["batch_size", "kv_cache_size"]],
        # )

        # Return all attention layer predictions / 返回注意力层所有预测结果
        return predictions

    def _predict_for_attention_layer_models(self) -> Dict[str, Any]:
        predictions = {}

        decode_batch_size_range = np.arange(
            1, self._config.prediction_max_batch_size + 1
        )
        decode_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,       # kv cache is stored in blocks, this is the block_size
        )
        decode_prefill_chunk_size_range = [0]
        # 生成所有的组合
        # Generate all combinations
        decode_batch_size, decode_kv_cache_size, decode_prefill_chunk_size = zip(
            *product(
                decode_batch_size_range,
                decode_kv_cache_size_range,
                decode_prefill_chunk_size_range,
            )
        )

        prefill_batch_size_range = [1]
        prefill_kv_cache_size_range = np.arange(
            0,
            self._config.prediction_max_tokens_per_request + 1,
            self._config.kv_cache_prediction_granularity,
        )
        prefill_prefill_chunk_size_range = np.arange(
            1, self._config.prediction_max_prefill_chunk_size + 1
        )
        prefill_batch_size, prefill_kv_cache_size, prefill_prefill_chunk_size = zip(
            *product(
                prefill_batch_size_range,
                prefill_kv_cache_size_range,
                prefill_prefill_chunk_size_range,
            )
        )

        attention_df = pd.DataFrame(
            {
                "batch_size": decode_batch_size + prefill_batch_size,
                "kv_cache_size": decode_kv_cache_size + prefill_kv_cache_size,
                "prefill_chunk_size": decode_prefill_chunk_size
                + prefill_prefill_chunk_size,
            }
        )

        attention_df["is_decode"] = attention_df["prefill_chunk_size"] == 0
        attention_df["num_tokens"] = attention_df[
            ["prefill_chunk_size", "batch_size"]
        ].max(axis=1)
        attention_df["prefill_chunk_size_squared"] = (
            attention_df["prefill_chunk_size"] ** 2
        )

        prefill_df = attention_df[~attention_df["is_decode"]]
        decode_df = attention_df[attention_df["is_decode"]]
        chunked_prefill_df = prefill_df[prefill_df["kv_cache_size"] > 0].copy()
        chunked_prefill_df["total_prefill_tokens"] = (
            chunked_prefill_df["kv_cache_size"]
            + chunked_prefill_df["prefill_chunk_size"]
        )

        # **2预测prefill attn时间
        # Predict prefill attention time using kv_cache_size and chunk_size**2
        predictions["attn_prefill"] = self._get_model_prediction(
            "attn_prefill",
            self._models["attn_prefill"],
            prefill_df[["kv_cache_size", "prefill_chunk_size_squared"]],
        )

        # 用kv_cache_size和batch_size预测decode attn时间
        # kv_cache_size是decode + prefill，即枚举上下文的长度
        # Predict decode attention time using kv_cache_size and batch_size
        # kv_cache_size is decode + prefill, i.e., enumerating context length
        predictions["attn_decode"] = self._get_model_prediction(
            "attn_decode",
            self._models["attn_decode"],
            decode_df[["batch_size", "kv_cache_size"]],
        )

        return predictions

    def _predict_from_models(self) -> Dict[str, Any]:
        predictions = self._predict_for_compute_models()    # All-reduce and send-recv are included here.
        predictions.update(self._predict_for_cpu_overhead_models())
        predictions.update(self._predict_for_attention_layer_models())

        return predictions
    
    def _predict_from_models_by_aicb(self) -> Dict[str, Any]:
        predictions = self._predict_for_compute_models_by_aicb()    # All-reduce and send-recv are included here.
        predictions.update(self._predict_for_cpu_overhead_models_by_aicb())
        predictions.update(self._predict_for_attention_layer_models_by_aicb())

        return predictions

    def _get_batch_decode_attention_params(self, batch: Batch) -> Tuple[int, int]:
        if hasattr(batch, "_decode_params"):
            return batch._decode_params

        decode_kv_cache_sizes = []

        for request in batch.requests:
            if request._is_prefill_complete:
                decode_kv_cache_sizes.append(request.num_processed_tokens)

        if not decode_kv_cache_sizes:
            batch._decode_params = (0, 0)
            return batch._decode_params

        decode_batch_size = len(decode_kv_cache_sizes)
        decode_avg_kv_cache_size = int(np.mean(decode_kv_cache_sizes))
        decode_avg_kv_cache_size = (
            (
                decode_avg_kv_cache_size
                + self._config.kv_cache_prediction_granularity
                - 1
            )
            // self._config.kv_cache_prediction_granularity
        ) * self._config.kv_cache_prediction_granularity

        batch._decode_params = (decode_batch_size, decode_avg_kv_cache_size)

        return batch._decode_params

    def _get_batch_prefill_attention_params(
        self, batch: Batch
    ) -> List[Tuple[int, int]]:
        if hasattr(batch, "_prefill_params"):
            return batch._prefill_params

        prefill_params = []

        for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
            if request._is_prefill_complete:
                continue

            prefill_chunk_size = num_tokens_to_process
            kv_cache_size = (
                (
                    request.num_processed_tokens
                    + self._config.kv_cache_prediction_granularity
                    - 1
                )
                // self._config.kv_cache_prediction_granularity
            ) * self._config.kv_cache_prediction_granularity

            prefill_params.append((kv_cache_size, prefill_chunk_size))

        batch._prefill_params = prefill_params

        return prefill_params

    # 线性层和激活层直接用total_num_tokens预测
    # Predict directly with total_num_tokens for the linear layer and activation layer.
    def _get_pipeline_protocol_time(self, batch: Batch, pipeline_stage: int, current_time):
        if self._config.skip_cpu_overhead_modeling or not getattr(self, '_pipeline_protocol_model_names', []):
            return None
        pp = self._replica_config.num_pipeline_stages
        predict = lambda name: self._get_cpu_profile_prediction(name, batch)
        if pipeline_stage == 0:
            batch._pipeline_protocol_elapsed_ms = 0.0
            begin = 0.0 if current_time is None else current_time
            launch = begin + (self._get_schedule_time(batch) + self._get_engine_bookkeeping_time(batch)
                              + predict('executor_pre_dispatch')) * 1e-3
            batch._pipeline_protocol_launch_at = launch
            if not hasattr(self, '_pipeline_worker_available_at'):
                self._pipeline_worker_available_at = [0.0] * pp
            batch._pipeline_input_ready_at = [
                max(launch + predict(f'pipeline_input_dispatch_stage_{s}') * 1e-3,
                    self._pipeline_worker_available_at[s])
                + predict(f'pipeline_input_work_stage_{s}') * 1e-3 for s in range(pp)]
            duration = (batch._pipeline_input_ready_at[0] - launch) * 1000 + predict('executor_pre_dispatch')
        else:
            if not hasattr(batch, '_pipeline_input_ready_at'):
                raise ValueError('Pipeline protocol must begin at stage zero')
            now = batch._pipeline_protocol_elapsed_ms * 1e-3 if current_time is None else current_time
            duration = max(0.0, (batch._pipeline_input_ready_at[pipeline_stage] - now) * 1000)
            duration += predict(f'pipeline_post_receive_stage_{pipeline_stage}')
        if pipeline_stage == pp - 1:
            duration += predict('pipeline_result_return')
        return duration

    def _get_eager_forward_execution_time(self, batch: Batch, pipeline_stage: int):
        if not batch.num_prefill_tokens or not getattr(self, '_eager_dispatch_programs', []):
            return None
        if self._config.backend != 'vidur':
            raise ValueError('CPU launch program requires independently profiled Vidur GPU primitives')
        if batch.num_decode_tokens:
            raise ValueError('Profile mixed prefill/decode CPU programs before simulating mixed batches')
        from vidur.utils.eager_dispatch_program import interpolate_program, replay_program
        program = interpolate_program(self._eager_dispatch_programs[pipeline_stage], batch.size,
                                      batch.num_prefill_tokens / batch.size)
        lo, hi = pipeline_layer_bounds(self._model_config.num_layers, pipeline_stage,
                                      self._replica_config.num_pipeline_stages)
        layers = {f['layer'] for f in program['frames'] if f['layer'] is not None}
        if layers != set(range(lo, hi)):
            raise ValueError('Eager CPU program does not match pipeline layer placement')
        tp = (self._get_tensor_parallel_communication_time(batch, pipeline_stage)
              if self._replica_config.tensor_parallel_size > 1 else 0.0)
        key = (self._get_compute_tokens(batch),)
        gpu = dict(embedding=self._predictions['emb'][key] + tp,
                   qkv=self._get_attention_layer_pre_proj_execution_time(batch),
                   rope=self._get_attention_rope_execution_time(batch),
                   attention=self._get_attention_kv_cache_save_execution_time(batch)
                             + self._get_attention_prefill_execution_time(batch),
                   attention_output=self._get_attention_layer_post_proj_execution_time(batch) + tp,
                   post_attention_norm=self._get_mlp_norm_layer_act_execution_time(batch),
                   mlp_up=self._get_mlp_layer_up_proj_execution_time(batch),
                   mlp_activation=self._get_mlp_layer_act_execution_time(batch),
                   mlp_down=self._get_mlp_layer_down_proj_execution_time(batch) + tp,
                   final_norm=self._predictions['final_layernorm'][key])
        def duration(operation, layer):
            if operation == 'input_norm':
                return (self._predictions['first_layernorm'][key] if layer == 0 else
                        self._get_attn_norm_layer_act_execution_time(batch))
            if operation not in gpu:
                raise ValueError(f'No GPU primitive for eager operation {operation}')
            return gpu[operation]
        return replay_program(program, duration)

    def _get_decoder_graph_execution_time(self, batch: Batch, pipeline_stage: int):
        if self._decoder_graph_profile is None or batch.num_prefill_tokens:
            return None
        from vidur.utils.runtime_layout import pipeline_layer_bounds
        lo, hi = pipeline_layer_bounds(self._model_config.num_layers, pipeline_stage,
                                       self._replica_config.num_pipeline_stages)
        kv = sum(r.num_processed_tokens for r in batch.requests) / batch.size
        return self._decoder_graph_profile.predict(hi - lo, batch.size, kv)

    def _get_compute_tokens(self, batch: Batch) -> int:
        if not getattr(self, "_attention_layout_aware", False):
            return batch._total_num_tokens_rounded
        count = sum(batch.num_tokens)
        if batch.num_prefill_tokens or self._config.decode_attention_execution_mode == "eager":
            return count
        max_context = max(r.num_processed_tokens + 1 for r in batch.requests)
        if max_context > self._attention_graph_capacity:
            raise ValueError("Decode exceeds CUDA graph capture coverage; profile the eager fallback before simulating it")
        from vidur.utils.runtime_layout import graph_batch_size
        return graph_batch_size(count)

    def _get_model_boundary_time(self, batch: Batch, pipeline_stage: int) -> float:
        if self._config.backend != "vidur":
            return 0.0
        key = (self._get_compute_tokens(batch),)
        total = 0.0
        if pipeline_stage == 0:
            total += self._predictions["emb"][key]
            if getattr(self, "_attention_layout_aware", False) and self._replica_config.tensor_parallel_size > 1:
                # Native VocabParallelEmbedding has one reduction, while its
                # standalone GPU target contains only local embedding work.
                total += self._get_tensor_parallel_communication_time(batch, pipeline_stage=pipeline_stage, include_boundary=True)
            if "first_layernorm" in self._predictions:
                # Replace one fused layer-input norm by the actual first norm.
                total += self._predictions["first_layernorm"][key] - self._predictions["input_layernorm"][key]
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            if "final_layernorm" in self._predictions:
                total += self._predictions["final_layernorm"][key]
        # Logits is already inside the profiled sampler_e2e interval.
        return total

    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["attn_pre_proj"][(self._get_compute_tokens(batch),)]

    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["attn_post_proj"][(self._get_compute_tokens(batch),)]

    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_up_proj"][(self._get_compute_tokens(batch),)]

    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_down_proj"][(self._get_compute_tokens(batch),)]

    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["mlp_act"][(self._get_compute_tokens(batch),)]

    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["input_layernorm"][(self._get_compute_tokens(batch),)]

    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        if not self._model_config.post_attn_norm:
            return 0

        return self._predictions["post_attention_layernorm"][
            (self._get_compute_tokens(batch),)
        ]

    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        return self._predictions["add"][(self._get_compute_tokens(batch),)]

    # profiling得到是单个allreduce在bytes上的时间
    # 模型训练得到的是(num_token -> 单次allreduce time)的映射 
    # Profiling gives the time of a single allreduce operation in bytes
    # Model training produces a mapping of (num_token -> single allreduce time)
    def _get_tensor_parallel_communication_time(self, batch: Batch, pipeline_stage: int = 0, include_boundary: bool = False) -> float:
        graph = getattr(self, "_decoder_graph_profile", None)
        if (graph is not None and graph.includes_tp_collectives
                and not batch.num_prefill_tokens and not include_boundary):
            tp = self._replica_config.tensor_parallel_size
            nodes = self._rank_node_map[pipeline_stage * tp:(pipeline_stage + 1) * tp]
            if len(nodes) != tp or len(set(nodes)) != 1:
                raise ValueError("Combined TP decoder graph lacks coverage for this rank placement")
            # This DAG contains both collectives per layer. Charge them once.
            return 0.0
        native = getattr(self, "_native_tp_collectives", {}).get(pipeline_stage)
        if native is not None:
            return native.predict(self._get_compute_tokens(batch),
                bool(batch.num_prefill_tokens) or self._config.decode_attention_execution_mode == "eager")
        names = getattr(self, "_tp_group_model_names", ["all_reduce"] * self._replica_config.num_pipeline_stages)
        name = names[pipeline_stage]
        if getattr(self, "_attention_layout_aware", False):
            # One CUDA graph replay launches the captured collectives; do not
            # add a per-layer Python launch or an arbitrary TP skew exponent.
            return self._predictions[name][(self._get_compute_tokens(batch),)]
        return (
            self._predictions[name][(self._get_compute_tokens(batch),)]
            + self._config.nccl_cpu_launch_overhead_ms
            + self._config.nccl_cpu_skew_overhead_per_device_ms
            * self._replica_config.tensor_parallel_size**1.25
            # TP group越大，all-reduce越慢，这里用**1.25来模拟这个效果
            # The larger the TP group, the slower the all-reduce, here we use **1.25 to simulate this effect
        )

    def _get_pipeline_parallel_communication_time(self, batch: Batch, pipeline_stage: int = 0) -> float:
        if getattr(self, "_has_pp_handoff_profile", False):
            names = getattr(self, "_pp_handoff_model_names", [])
            if names:
                return self._get_cpu_profile_prediction(names[pipeline_stage], batch)
            # The measured executor residual excludes this complete tensor_dict
            # handoff. Add it once across all non-final boundaries, not again
            # as a bare NCCL send_recv prediction.
            return self._get_cpu_profile_prediction("pp_handoff_e2e", batch) / (
                self._replica_config.num_pipeline_stages - 1)
        return self._predictions["send_recv"][(self._get_compute_tokens(batch),)]

    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        if self._config.backend != "vidur":
            return 0
        if self._model_config.rope_theta is None:
            return 0
        return self._predictions["attn_rope"][(self._get_compute_tokens(batch),)]

    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        # don't use round up to the nearest multiple of 8 here, because we want to
        # predict the execution time for the exact number of tokens
        num_tokens = sum(batch.num_tokens)

        return self._predictions["attn_kv_cache_save"][(num_tokens,)]

    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        (
            decode_batch_size,
            decode_avg_kv_cache_size,      # Average context length per request
        ) = self._get_batch_decode_attention_params(batch)
        if decode_batch_size == 0:
            return 0
        # TODO(tianhao909): add decode output logging
        # TODO(tianhao909): 添加 decode 输出日志
        value = self._predictions["attn_decode"][(decode_batch_size, decode_avg_kv_cache_size)]
        if getattr(self, "_attention_layout_aware", False):
            return value
        return value * (
            1
            + self._attention_decode_batching_overhead_fraction
            * int(decode_batch_size > 1)
        )

    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        prefill_params = self._get_batch_prefill_attention_params(batch)

        if len(prefill_params) == 0:
            return 0

        kv_cache_sizes, prefill_chunk_sizes = zip(*prefill_params)
        logger.debug(f"prefill_chunk_sizes length: {len(prefill_chunk_sizes)}")

        agg_kv_cache_size = sum(kv_cache_sizes)
        agg_prefill_chunk_size = sum([x**2 for x in prefill_chunk_sizes]) ** 0.5

        value = self._predictions["attn_prefill"][
            (agg_kv_cache_size, round(agg_prefill_chunk_size) ** 2)]
        if getattr(self, "_attention_layout_aware", False):
            return value
        return value * (
            1 + self._attention_prefill_batching_overhead_fraction * int(len(prefill_params) > 1))

    def _get_graph_input_staging_time(self, batch: Batch, pipeline_stage: int) -> float:
        if self._config.skip_cpu_overhead_modeling or not getattr(self, "_graph_staging_model_names", []):
            return 0.0
        return self._get_cpu_profile_prediction(self._graph_staging_model_names[pipeline_stage], batch)

    def _get_engine_bookkeeping_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling or not getattr(self, "_has_engine_bookkeeping_profile", False):
            return 0.0
        return self._get_cpu_profile_prediction("engine_bookkeeping", batch)

    def _get_schedule_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0
        return self._get_cpu_profile_prediction("schedule", batch)

    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0
        return self._get_cpu_profile_prediction("sampler_e2e", batch)

    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0
        return self._get_cpu_profile_prediction("prepare_inputs_e2e", batch)

    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0
        return self._get_cpu_profile_prediction("process_model_outputs", batch)

    def _get_ray_comm_time(self, batch: Batch) -> float:
        if self._config.skip_cpu_overhead_modeling:
            return 0
        return self._get_cpu_profile_prediction("ray_comm_time", batch)

    def to_dict(self) -> dict:
        return {
            "model_provider": str(self._config.get_type()),
            "num_tensor_parallel_workers": self._replica_config.tensor_parallel_size,
            "k_fold_cv_splits": self._config.k_fold_cv_splits,
            "num_q_heads": self._model_config.num_q_heads,
            "num_kv_heads": self._model_config.num_kv_heads,
            "embedding_dim": self._model_config.embedding_dim,
            "mlp_hidden_dim": self._model_config.mlp_hidden_dim,
            "use_gated_mlp": self._model_config.use_gated_mlp,
            "vocab_size": self._model_config.vocab_size,
            "block_size": self._block_size,
            "max_tokens": self._max_tokens,
            "compute_input_file": self._compute_input_file,
            "all_reduce_input_file": self._all_reduce_input_file,
            "send_recv_input_file": self._send_recv_input_file,
            "cpu_overhead_input_file": self._cpu_overhead_input_file,
            "network_transport": self._config.network_transport,
            "prediction_max_prefill_chunk_size": self._config.prediction_max_prefill_chunk_size,
            "max_batch_size": self._config.prediction_max_batch_size,
        }
