"""Retain existing graph routing buffers; copy only during diagnostic requests.

Startup retains references without adding GPU work to captured graphs. Timed
control requests don't copy routes. Diagnostic snapshots copy on the original
stream after the model event and before the next graph replay can overwrite it.
"""
import functools,json,pathlib,os
import torch
from alignment_worker import AlignmentWorker
from qwen_audit_worker import ROOT
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

_LAYER=None
_WORKER=None
_DECODE_REFS={}
_original_forward=FusedMoE.forward_impl
_original_select=FusedMoE.select_experts

@functools.wraps(_original_forward)
def _forward(layer,*args,**kwargs):
    global _LAYER
    previous=_LAYER;_LAYER=layer.layer_name
    try:return _original_forward(layer,*args,**kwargs)
    finally:_LAYER=previous

@functools.wraps(_original_select)
def _select(*args,**kwargs):
    result=_original_select(*args,**kwargs)
    ids=result[1]
    if _LAYER is not None and torch.cuda.is_current_stream_capturing() and ids.shape[0]==4:
        _DECODE_REFS[_LAYER]=ids.detach()
    worker=_WORKER
    if worker is not None and getattr(worker,'_capture_prefill_now',False):
        worker._route_rows.append(dict(phase='prefill',layer=_LAYER,ids=ids.detach(),request=worker._route_request,step=worker._route_prefill_index))
    return result

FusedMoE.forward_impl=_forward
FusedMoE.select_experts=staticmethod(_select)

class _RouteProxy:
    def __init__(self,model,worker):self._model,self._worker=model,worker
    def __getattr__(self,name):return getattr(self._model,name)
    def __call__(self,*args,**kwargs):
        w=self._worker;it=getattr(w,'_q80_current_iteration',None)
        active=bool(it and it.get('request_tokens'))
        prefill=active and it['total_tokens']>1
        if active:
            request=next(iter(it['request_tokens']))
            if request!=w._route_request:w._route_request=request;w._route_decode_index=0;w._route_prefill_index=0
            if prefill:w._route_prefill_index+=1
            else:w._route_decode_index+=1
        w._capture_prefill_now=prefill and w._route_prefill_index==1
        try:result=self._model(*args,**kwargs)
        finally:w._capture_prefill_now=False
        if active and not prefill and w._route_decode_index in (1,8,16,32,64,128,255):
            for name,ids in _DECODE_REFS.items():
                w._route_rows.append(dict(phase='decode',layer=name,ids=ids.clone(),request=w._route_request,step=w._route_decode_index))
        return result

class RoutingWorker(AlignmentWorker):
    def q80_runtime_config(self):
        config=self.vllm_config
        gdn=[m for m in self.get_model().modules() if type(m).__name__=='Qwen3NextGatedDeltaNet']
        return dict(**self.q80_alignment_info(),block_size=config.cache_config.block_size,
                    decode_route_buffers=len(_DECODE_REFS),
                    gdn_cache=[dict(name=m.prefix,state=[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in m.kv_cache[0]]) for m in gdn[:1]])

    def q80_start_routes(self):
        global _WORKER
        self._route_rows=[];self._capture_prefill_now=False;self._route_request=None;self._route_decode_index=0;self._route_prefill_index=0
        _WORKER=self
        self.q80_start_timers();self.model_runner.model=_RouteProxy(self.model_runner.model,self)
        return self.q80_runtime_config()

    def q80_stop_routes(self,label):
        global _WORKER
        self._capture_prefill_now=False;_WORKER=None
        self.q80_stop_timers(label)
        rank=self.vllm_config.parallel_config.data_parallel_rank
        d=ROOT/'route_corpus'/label;d.mkdir(parents=True,exist_ok=True)
        rows=[];summary=[]
        for item in self._route_rows:
            row=dict(item);row['ids']=row['ids'].cpu();rows.append(row)
            histogram=torch.bincount(row['ids'].flatten().to(torch.int64),minlength=512).tolist()
            summary.append(dict(phase=row['phase'],layer=row['layer'],request=row['request'],step=row['step'],shape=list(row['ids'].shape),histogram=histogram))
        path=d/f'dp{rank}.pt';torch.save(rows,path)
        (d/f'dp{rank}.json').write_text(json.dumps(dict(info=self.q80_runtime_config(),rows=summary)))
        self._route_rows=[]
        return dict(rank=rank,rows=len(rows),path=str(path))
