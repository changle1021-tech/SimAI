"""Diagnostic methods around opaque vLLM custom-op bodies, installed by RPC.

No NN forward hooks are added to Dynamo-traced outer model/layer functions.
Existing decode CUDA Graphs remain unchanged; component wrappers observe
prefill Python execution only. CPU-only and event-timed passes are separate.
"""
import functools,json,pathlib,time,inspect
import torch
from qwen_audit_worker import AuditWorker,ROOT


class AlignmentWorker(AuditWorker):
    def q80_alignment_info(self):
        runner=self.model_runner
        from vllm import envs
        moes=[m for m in self.get_model().modules() if type(m).__name__=='SharedFusedMoE']
        return dict(**self.q80_info(),runner_methods=[n for n in dir(runner) if any(k in n for k in ('prepare','preprocess','sample','update','logits'))],
                    quant_methods=sorted({str(type(m.quant_method)) for m in moes}),
                    shared_stream_disabled=envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM,
                    shared_threshold=envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD,
                    moe_layers=[dict(name=m.layer_name,tp=m.tp_size,dp=m.dp_size,ep=m.ep_size,local_experts=m.local_num_experts) for m in moes])

    def q80_start_components(self,gpu_events=False):
        if getattr(self,'_align_originals',None):raise RuntimeError('component timers already installed')
        self.q80_start_timers()
        self._align_rows=[];self._align_originals=[];self._align_gpu_events=gpu_events
        self._align_current_layer=None
        worker=self

        def call(kind,fn,obj,args,kwargs):
            iteration=getattr(worker,'_q80_current_iteration',None)
            # CUDA Graph replay bypasses these Python custom-op bodies. Ignore
            # dummy forwards and any remaining runner activity after execute.
            if not iteration:return fn(obj,*args,**kwargs)
            layer=getattr(obj,'layer_name',getattr(obj,'prefix',None))
            if kind=='experts' and kwargs.get('layer') is not None:layer=kwargs['layer'].layer_name
            layer=layer or worker._align_current_layer
            old=worker._align_current_layer
            if kind=='moe':worker._align_current_layer=layer
            start_cpu=time.perf_counter_ns();start=end=None
            if gpu_events:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
            try:return fn(obj,*args,**kwargs)
            finally:
                if end is not None:end.record()
                worker._align_rows.append(dict(kind=kind,layer=layer,cpu_start_ns=start_cpu,cpu_end_ns=time.perf_counter_ns(),start=start,end=end,iteration=iteration))
                worker._align_current_layer=old

        def patch(cls,name,kind):
            if not hasattr(cls,name):return
            original=getattr(cls,name)
            @functools.wraps(original)
            def wrapped(obj,*args,**kwargs):return call(kind,original,obj,args,kwargs)
            worker._align_originals.append((cls,name,original));setattr(cls,name,wrapped)

        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet,Qwen3NextMLP,Qwen3NextForCausalLM
        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        patch(FusedMoE,'forward_impl','moe')
        patch(Qwen3NextGatedDeltaNet,'_forward_core','gdn_core_chain')
        patch(Qwen3NextMLP,'forward','shared_expert')
        patch(AgRsAll2AllManager,'dispatch','dispatch')
        patch(AgRsAll2AllManager,'combine','combine')
        patch(FlashAttentionImpl,'forward','flash_attention')
        patch(Qwen3NextForCausalLM,'compute_logits','lm_head')
        qclasses={type(m.quant_method) for m in self.get_model().modules() if type(m).__name__=='SharedFusedMoE'}
        for cls in qclasses:patch(cls,'apply','experts')
        runner_cls=type(self.model_runner)
        for name in ['_prepare_inputs','_update_states','_preprocess','_sample','_bookkeeping_sync','_prepare_sampling','_prepare_attention_metadata']:
            patch(runner_cls,name,'runner'+name)
        return dict(dp_rank=self.vllm_config.parallel_config.data_parallel_rank,gpu_events=gpu_events,patches=[f'{c.__name__}.{n}' for c,n,f in self._align_originals])

    def q80_stop_components(self,label):
        for cls,name,original in reversed(self._align_originals):setattr(cls,name,original)
        self._align_originals=[]
        self.q80_stop_timers(label)
        torch.cuda.synchronize()
        rows=[]
        for row in self._align_rows:
            start,end=row.pop('start'),row.pop('end')
            row['cpu_ms']=(row['cpu_end_ns']-row['cpu_start_ns'])/1e6
            if start is not None:row['gpu_ms']=start.elapsed_time(end)
            rows.append(row)
        rank=self.vllm_config.parallel_config.data_parallel_rank
        path=ROOT/'components'/label/f'dp{rank}.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(info=self.q80_alignment_info(),gpu_events=self._align_gpu_events,rows=rows)))
        return dict(dp_rank=rank,rows=len(rows),path=str(path))
