"""Matched coarse TP execution profiles, separate from legacy kernel tables.

Uses means of measured samples at each shape and bounded linear interpolation
within that shape's measured axis. It never learns from request E2E errors.
"""
import csv
import hashlib
import math
from pathlib import Path
import statistics
from bisect import bisect_left

COMPONENTS = ('schedule_ms', 'input_prefix_ms', 'model_gpu_ms',
              'post_model_sampler_ms', 'executor_return_ms', 'output_processing_ms')
SCOPE = 'async_driver_gpu_and_critical_wall_v1'

class AsyncExecutionProfile:
    def __init__(self, path, model_config, replica_config):
        if replica_config.num_pipeline_stages != 1:
            raise ValueError('This TP-step profile has no measured per-stage PP dependency boundaries.')
        path = Path(path)
        payload = path.read_bytes()
        self.source_path = str(path.resolve())
        self.source_sha256 = hashlib.sha256(payload).hexdigest()
        required = {'timing_scope','profile_group','model_name','num_layers','n_embd','n_expanded_embd',
                    'n_q_head','n_kv_head','vocab_size','num_tensor_parallel_workers','num_pipeline_stages',
                    'phase','batch_size','prefill_tokens','sequence_length','forward_tokens','cuda_graph',*COMPONENTS}
        with path.open(newline='') as handle:
            reader = csv.DictReader(handle)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Incomplete matched profile columns: '+str(sorted(required-set(reader.fieldnames or []))))
            source_rows = list(reader)
        dimensions = {'num_layers':model_config.num_layers,'n_embd':model_config.embedding_dim,
                      'n_expanded_embd':model_config.mlp_hidden_dim,'n_q_head':model_config.num_q_heads,
                      'n_kv_head':model_config.num_kv_heads,'vocab_size':model_config.vocab_size,
                      'num_tensor_parallel_workers':replica_config.tensor_parallel_size,
                      'num_pipeline_stages':replica_config.num_pipeline_stages}
        selected = [r for r in source_rows if r['model_name']==model_config.get_name()
                    and all(int(r[k])==v for k,v in dimensions.items())]
        if not selected:
            raise ValueError('No measured profile matches the model structure and TP configuration.')
        if {r['timing_scope'] for r in selected}!={SCOPE} or len({r['profile_group'] for r in selected})!=1:
            raise ValueError('Mixed timing scopes or measurement versions in selected profile.')
        self.profile_group = selected[0]['profile_group']
        groups = {}
        for r in selected:
            phase = r['phase'];bs = int(r['batch_size'])
            graph = r['cuda_graph'].lower()=='true'
            if phase not in ('prefill','decode') or bs < 1 or graph != (phase=='decode'):
                raise ValueError('Unmatched eager-prefill/graph-decode execution condition.')
            if int(r['forward_tokens']) < bs:
                raise ValueError('Model-forward shape smaller than actual request batch.')
            x = float(r['prefill_tokens'] if phase=='prefill' else r['sequence_length'])
            if not math.isfinite(x) or x <= 0:
                raise ValueError('Invalid measured shape axis.')
            if phase=='prefill' and abs(float(r['sequence_length'])*bs-x)>1e-6:
                raise ValueError('This first-prefill profile contains cached-token work.')
            costs = {k:float(r[k]) for k in COMPONENTS}
            if any(not math.isfinite(v) or v<0 for v in costs.values()) or costs['model_gpu_ms']<=0:
                raise ValueError('Negative/nonfinite measured profile component.')
            groups.setdefault((phase,bs),{}).setdefault(x,[]).append(costs)
        self.points = {key:{x:{c:statistics.fmean(row[c] for row in rows) for c in COMPONENTS}
                            for x,rows in values.items()} for key,values in groups.items()}
        self.axes = {key:sorted(points) for key,points in self.points.items()}

    def predict(self, batch):
        if batch.num_prefill_tokens and batch.num_decode_tokens:
            raise ValueError('The selected profile does not cover mixed prefill/decode batches.')
        phase = 'prefill' if batch.num_prefill_tokens else 'decode'
        if phase=='prefill':
            if any(r.num_processed_tokens for r in batch.requests):
                raise ValueError('The selected first-prefill profile does not cover cached/chunked prefill.')
            x = float(batch.num_prefill_tokens)
        else:
            x = statistics.fmean(r.num_processed_tokens for r in batch.requests)
        key = (phase,batch.size)
        if key not in self.axes:
            raise ValueError('Unmeasured phase/batch size in selected profile: '+str(key))
        axis=self.axes[key];points=self.points[key]
        if x<axis[0] or x>axis[-1]:
            raise ValueError('Requested shape outside measured profile range: '+str((key,x,axis[0],axis[-1])))
        i=bisect_left(axis,x)
        if axis[i]==x:
            return dict(points[x])
        lo,hi=axis[i-1],axis[i];fraction=(x-lo)/(hi-lo)
        return {c:points[lo][c]+fraction*(points[hi][c]-points[lo][c]) for c in COMPONENTS}
