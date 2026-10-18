"""Validated Qwen profiles. No nearest-neighbor or cross-model fallbacks.

Interpolation uses measured files on disk rather than the history of previous
requests. CSV payloads and producer source identity are checked before use.
This module deliberately has no CUDA, torch, Ray, or simulator imports.
"""
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path


def producer_identity(aicb_root):
    folder = Path(aicb_root) / "workload_generator/mocked_model/inference"
    return hashlib.sha256(b"".join((folder / n).read_bytes() for n in (
        "qwen3_next_projections.py", "AiobQwen3Next.py",
        "qwen3_next_attention.py", "qwen3_next_moe.py"))).hexdigest()


def interpolate(low, high, alpha):
    if set(low) != set(high):
        raise ValueError("Qwen profile layer layouts differ")
    result = {}
    for layer, parts in low.items():
        if set(parts) != set(high[layer]):
            raise ValueError("Qwen profile component layouts differ")
        if any(parts[name].get("includes_ep_comm",0) != high[layer][name].get("includes_ep_comm",0) for name in parts):
            raise ValueError("Cannot interpolate Qwen profiles with different MoE timing scopes")
        result[layer] = {name: {
            key: value * (1-alpha) + high[layer][name][key] * alpha
            for key, value in fields.items()} for name, fields in parts.items()}
    return result


def bracket(points, value):
    points = sorted(points)
    if value in points:
        return value, value
    index = bisect.bisect_left(points, value)
    if index == 0 or index == len(points):
        raise ValueError("Qwen profile query {} is outside measured range {}".format(value, points))
    return points[index-1], points[index]


class QwenAICBProfiles:
    def __init__(self, aicb_root, profile_dir=None):
        self.root = Path(aicb_root)
        self.directory = Path(profile_dir) if profile_dir else self.root / "results/workload"
        self.source = producer_identity(self.root)
        self._validated = {}

    def _read(self, path, expected):
        path = Path(path)
        sidecar = Path(str(path)+".projections.json")
        payload = path.read_bytes()
        metadata = json.loads(sidecar.read_text())
        identity = hashlib.sha256(payload+sidecar.read_bytes()).hexdigest()
        if metadata.get("projection_profile_version") != 3 or metadata.get("scope") != "qwen3_next_components":
            raise ValueError("Legacy Qwen AICB profile: regenerate {}".format(path))
        if metadata.get("source_sha256") != self.source:
            raise ValueError("Qwen producer source changed: regenerate {}".format(path))
        if metadata.get("profile_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("Qwen profile payload hash differs: {}".format(path))
        if metadata.get("precision") != "bf16":
            raise ValueError("BF16 Qwen execution cannot consume an FP8 component profile")
        invocation = metadata.get("invocation", {})
        for key, value in expected.items():
            if invocation.get(key) != value:
                raise ValueError("Qwen profile {} mismatch for {}: {} != {}".format(path, key, invocation.get(key), value))
        if identity in self._validated:
            return self._validated[identity]
        rows = csv.DictReader(payload.decode().splitlines(), delimiter="\t")
        result = {}
        for row in rows:
            layer = int(row["layer_id"])
            name = row["layer_name"]
            if name in result.setdefault(layer, {}):
                raise ValueError("Duplicate Qwen profile component: {}/{}".format(layer, name))
            values = {key: float(row[key]) for key in ("comp_time", "comm_size")}
            if name == "moe":
                scope = metadata.get("moe_time_scope", "compute_only")
                if scope not in ("compute_only", "compute_and_ep_comm"):
                    raise ValueError("Unknown Qwen MoE timing scope")
                values["includes_ep_comm"] = float(scope == "compute_and_ep_comm")
            if any(not math.isfinite(v) or v < 0 for v in values.values()):
                raise ValueError("Nonfinite or negative Qwen timing/traffic")
            result[layer][name] = values
        if set(result) != set(range(48)) or any(set(p) != {"attention", "moe"} for p in result.values()):
            raise ValueError("Qwen3-Next-80B requires all 48 attention/MoE layer rows")
        if any(p[n]["comp_time"] <= 0 for p in result.values() for n in p):
            raise ValueError("Qwen component profile contains zero compute time")
        self._validated[identity] = result
        return result

    def load(self, ws, tp, pp, ep, bs, seq, phase, context=0):
        if bs != 1 or pp != 1:
            raise ValueError("Validated Qwen AICB profiles currently require batch1 and PP1")
        prefix = "vidur-Qwen3-Next-80B-world_size{}-tp{}-pp{}-ep{}-bs{}-seq".format(ws,tp,pp,ep,bs)
        candidates = {}
        for path in self.directory.glob(prefix+"*-"+phase+".csv"):
            sidecar = Path(str(path)+".projections.json")
            if not sidecar.is_file():
                continue
            inv = json.loads(sidecar.read_text()).get("invocation", {})
            tokens = inv.get("seq_length")
            ctx = inv.get("context_length", 0) if phase == "prefill" else 0
            if isinstance(tokens, int) and isinstance(ctx, int):
                point = tokens, ctx
                if point in candidates:
                    raise ValueError("Ambiguous Qwen profiles at {}".format(point))
                candidates[point] = path
        if not candidates:
            raise FileNotFoundError("No validated Qwen AICB profiles in {}".format(self.directory))
        expected = dict(model_name="Qwen3-Next-80B", world_size=ws,
                        tensor_model_parallel_size=tp,pipeline_model_parallel=pp,
                        expert_model_parallel_size=ep,micro_batch=bs,phase=phase)

        def read_point(tokens, ctx):
            return self._read(candidates[(tokens,ctx)], dict(expected,seq_length=tokens,context_length=ctx))

        if phase == "decode":
            lo, hi = bracket([p[0] for p in candidates if p[1]==0], seq)
            a=read_point(lo,0)
            return a if lo==hi else interpolate(a,read_point(hi,0),(seq-lo)/(hi-lo))
        if phase != "prefill":
            raise ValueError("Unknown Qwen profile phase")
        # Interpolate each chunk size along the existing-context axis first.
        # Only token sizes that bracket this context participate.
        by_tokens = {}
        for tokens in sorted({p[0] for p in candidates}):
            points = [p[1] for p in candidates if p[0]==tokens]
            if min(points) <= context <= max(points):
                by_tokens[tokens] = points
        lo,hi=bracket(by_tokens,seq)
        def read_tokens(tokens):
            a,b=bracket(by_tokens[tokens],context)
            low=read_point(tokens,a)
            return low if a==b else interpolate(low,read_point(tokens,b),(context-a)/(b-a))
        a=read_tokens(lo)
        return a if lo==hi else interpolate(a,read_tokens(hi),(seq-lo)/(hi-lo))


class RuntimeBoundaryProfile:
    """Net execution perimeter; includes waits/GPU work outside model.forward.

    These timings must not be summed with CPU function wall times that already
    wait for the model. They replace the old five empirical overhead formulas.
    """
    def __init__(self, path):
        self.data=json.loads(Path(path).read_text())
        schemas = {"qwen_worker_critical_perimeter_v2": "host_iteration_wall_minus_model_gpu",
                   "qwen_worker_request_perimeter_v3": "host_iteration_wall_minus_model_gpu_plus_request_io"}
        schema = self.data.get("schema")
        if schema not in schemas:
            raise ValueError("Unsupported worker perimeter profile")
        if self.data.get("scope") != schemas[schema]:
            raise ValueError("Worker perimeter profile has incompatible timing boundary")

    def estimate(self, phase, tokens, ws, tp, pp, ep, context=0):
        expected=dict(model="qwen3-next-80B",world_size=ws,tp=tp,pp=pp,ep=ep,batch_size=1)
        if self.data.get("configuration") != expected:
            raise ValueError("Worker perimeter profile configuration differs")
        if phase not in ("prefill", "decode"):
            raise ValueError("Unknown worker perimeter phase")
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            raise ValueError("Worker perimeter context must be a non-negative integer")
        extra = 0.0
        key = phase
        if self.data["schema"] == "qwen_worker_request_perimeter_v3" and phase == "prefill":
            key = "prefill_first" if context == 0 else "prefill_next"
            extra = self.data["request_io_ms"] if context == 0 else self.data["inter_chunk_gap_ms"]
        elif self.data["schema"] == "qwen_worker_request_perimeter_v3":
            extra = self.data["inter_decode_gap_ms"]
        if not math.isfinite(extra) or extra < 0:
            raise ValueError("Invalid request IO or inter-iteration time")
        points=self.data.get("phases",{}).get(key,[])
        by_tokens={p['tokens']:p for p in points}
        lo,hi=bracket(by_tokens,tokens)
        def cost(k):
            p=by_tokens[k]
            values=[p['critical_perimeter_ms']]
            if any(not math.isfinite(v) or v<0 for v in values):
                raise ValueError("Invalid worker perimeter time")
            return sum(values)
        a=cost(lo)
        return extra + (a if lo==hi else a+(cost(hi)-a)*(tokens-lo)/(hi-lo))


class RuntimeMoECommunication:
    """Phase-aware, max-rank AG(hidden,router)+RS(hidden) measurements."""
    def __init__(self, path):
        self.data=json.loads(Path(path).read_text())
        if self.data.get("world_size") != 4 or self.data.get("dtype") != "bfloat16":
            raise ValueError("MoE communication profile must be four-rank BF16 AG/RS")

    def estimate(self, phase, tokens, ws, tp, pp, ep):
        if (ws,tp,pp,ep) != (4,1,1,4):
            raise ValueError("Measured Qwen communication requires WS4 TP1 PP1 EP4")
        mode="eager" if phase=="prefill" else "cuda_graph"
        points={p['active_tokens']:p['median_ms'] for p in self.data['profiles']
                if p['mode']==mode and p['sizes']==[p['active_tokens']]*4}
        lo,hi=bracket(points,tokens)
        if any(not math.isfinite(v) or v<=0 for v in points.values()):
            raise ValueError("Invalid MoE communication timing")
        a=points[lo]
        return a if lo==hi else a+(points[hi]-a)*(tokens-lo)/(hi-lo)
