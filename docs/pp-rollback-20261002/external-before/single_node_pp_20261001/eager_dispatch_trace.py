"""CPU launch tape for the native Llama eager forward; no GPU waits/events."""
import time


def install_llama_dispatch_trace(worker, model):
    body = getattr(model, 'model', None)
    if body is None or not hasattr(body, 'layers'):
        raise ValueError('No eager dispatch adapter for this model architecture')
    wrapped={}

    def wrap(module, operation, layer=None):
        if id(module) in wrapped:
            if wrapped[id(module)]!=operation:
                raise ValueError('A shared module has different eager operation roles')
            return
        wrapped[id(module)]=operation
        original = module.forward
        def traced(*args, **kwargs):
            row = worker._pp_profile_current
            if row is None or not row.get('_capture_cpu_dispatch', False):
                return original(*args, **kwargs)
            start = time.perf_counter_ns()
            cpu = time.thread_time_ns()
            actual_layer=row.get('_dispatch_layer',layer) if layer is not None else None
            try:
                return original(*args, **kwargs)
            finally:
                end = time.perf_counter_ns()
                row.setdefault('cpu_dispatch_frames', []).append(dict(
                    operation=operation, layer=actual_layer, start_ns=start, end_ns=end,
                    thread_cpu_ms=(time.thread_time_ns()-cpu)*1e-6))
        module.forward = traced

    if worker.rank // worker.parallel_config.tensor_parallel_size == 0:
        wrap(body.embed_tokens, 'embedding')
    for index, layer in enumerate(body.layers):
        # vLLM retains placeholder modules for layers assigned to other stages.
        if not hasattr(layer, 'self_attn'):
            continue
        # get_rope caches one rotary module shared by many decoder layers.
        # Hook shared leaves once and identify the caller through this scope.
        original_layer=layer.forward
        def scoped_layer(*args,_original=original_layer,_index=index,**kwargs):
            row=worker._pp_profile_current
            if row is None or not row.get('_capture_cpu_dispatch',False):
                return _original(*args,**kwargs)
            previous=row.get('_dispatch_layer')
            row['_dispatch_layer']=_index
            try:
                return _original(*args,**kwargs)
            finally:
                if previous is None:row.pop('_dispatch_layer',None)
                else:row['_dispatch_layer']=previous
        layer.forward=scoped_layer
        wrap(layer.input_layernorm, 'input_norm', index)
        wrap(layer.self_attn.qkv_proj, 'qkv', index)
        wrap(layer.self_attn.rotary_emb, 'rope', index)
        wrap(layer.self_attn.attn, 'attention', index)
        wrap(layer.self_attn.o_proj, 'attention_output', index)
        wrap(layer.post_attention_layernorm, 'post_attention_norm', index)
        wrap(layer.mlp.gate_up_proj, 'mlp_up', index)
        wrap(layer.mlp.act_fn, 'mlp_activation', index)
        wrap(layer.mlp.down_proj, 'mlp_down', index)
    if worker.rank // worker.parallel_config.tensor_parallel_size == worker.parallel_config.pipeline_parallel_size-1:
        wrap(body.norm, 'final_norm')


def dispatch_program(row):
    """Convert non-nested CPU leaves to a producer timeline without E2E data."""
    previous = row['cpu_forward_start_ns']
    frames = []
    for frame in row['cpu_dispatch_frames']:
        start, end = frame['start_ns'], frame['end_ns']
        if not previous <= start <= end <= row['cpu_forward_end_ns']:
            raise ValueError('Eager dispatch leaves overlap or contradict root timing')
        frames.append(dict(operation=frame['operation'], layer=frame['layer'],
                           gap_ms=(start-previous)*1e-6,
                           call_ms=(end-start)*1e-6))
        previous = end
    if not frames:
        raise ValueError('Empty eager dispatch program')
    return dict(frames=frames, tail_ms=(row['cpu_forward_end_ns']-previous)*1e-6)


def mean_program(programs):
    if not programs:
        raise ValueError('Missing eager dispatch programs')
    signature = [(f['operation'], f['layer']) for f in programs[0]['frames']]
    if any([(f['operation'], f['layer']) for f in p['frames']] != signature for p in programs):
        raise ValueError('Cannot combine different eager execution programs')
    count = len(programs)
    return dict(frames=[dict(operation=op, layer=layer,
                           gap_ms=sum(p['frames'][i]['gap_ms'] for p in programs)/count,
                           call_ms=sum(p['frames'][i]['call_ms'] for p in programs)/count)
                       for i, (op, layer) in enumerate(signature)],
                tail_ms=sum(p['tail_ms'] for p in programs)/count)
