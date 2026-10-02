"""Native decoder DAG GPU spans; no client or engine E2E target is used."""
import argparse,os,json,math,statistics,gc
from pathlib import Path
from vllm_051_profile_common import init_tp,close_tp,check_vllm_version,run_all_tp,write_csv,put_stats
p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--model-name-for-vidur');p.add_argument('--tensor-parallel-sizes',type=int,nargs='+',default=[1,2])
p.add_argument('--layers',type=int,nargs='+',default=[8,16,32]);p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,4,8]);p.add_argument('--kv-sizes',type=int,nargs='+',default=[512,576,1024]);p.add_argument('--capacity',type=int,default=4096)
p.add_argument('--include-tp-collectives',action='store_true',help='Capture the native TP group inside the decoder DAG')
p.add_argument('--warmup',type=int,default=10);p.add_argument('--repetitions',type=int,default=20);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--part-output')
from native_attention_geometry import local_attention_heads
a=p.parse_args();check_vllm_version()
if not a.worker:
 forwarded=['--model',a.model,'--layers',*map(str,a.layers),'--batch-sizes',*map(str,a.batch_sizes),'--kv-sizes',*map(str,a.kv_sizes),'--capacity',str(a.capacity),'--warmup',str(a.warmup),'--repetitions',str(a.repetitions),'--output',a.output]
 if a.model_name_for_vidur:forwarded+=['--model-name-for-vidur',a.model_name_for_vidur]
 if a.include_tp_collectives:forwarded.append('--include-tp-collectives')
 run_all_tp(str(Path(__file__).resolve()),forwarded,a.tensor_parallel_sizes,a.output)
 raise SystemExit
import torch
from transformers import AutoConfig
from vllm.model_executor.models.llama import LlamaDecoderLayer
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.distributed.parallel_state import set_custom_all_reduce
if a.include_tp_collectives:set_custom_all_reduce(False)
rank,local_rank,tp=init_tp();config=AutoConfig.from_pretrained(a.model);torch.set_default_dtype(torch.float16)
if config.model_type!='llama' or config.hidden_act!='silu':raise ValueError('This adapter supports Llama/SILU; other architectures need their native adapter')
layers=[]
for _ in range(max(a.layers)):
 layer=LlamaDecoderLayer(config).cuda().eval();initialize_dummy_weights(layer)
 layer.self_attn.o_proj.reduce_results=a.include_tp_collectives;layer.mlp.down_proj.reduce_results=a.include_tp_collectives;layers.append(layer)
rows=[]
try:
 with torch.inference_mode():
  for count in a.layers:
   for batch in a.batch_sizes:
    physical=batch if batch<=2 else 4 if batch<=4 else math.ceil(batch/8)*8
    blocks=math.ceil(a.capacity/16);_,heads=local_attention_heads(config.num_attention_heads,config.num_key_value_heads,tp);dim=config.hidden_size//config.num_attention_heads
    cache=[torch.zeros((2,physical*blocks,16,heads,dim),device='cuda') for _ in range(count)]
    for kv in a.kv_sizes:
     if kv+1>a.capacity:raise ValueError('KV exceeds capture coverage')
     lengths=[kv+1]*batch+[1]*(physical-batch)
     tables=torch.arange(physical*blocks,device='cuda',dtype=torch.int32).view(physical,blocks)
     slots=torch.arange(physical,device='cuda',dtype=torch.int64)*blocks*16+kv;slots[batch:]=-1
     metadata=FlashAttentionMetadata(num_prefills=0,num_prefill_tokens=0,num_decode_tokens=physical,
      slot_mapping=slots,seq_lens=lengths,seq_lens_tensor=torch.tensor(lengths,device='cuda',dtype=torch.int32),
      max_query_len=None,max_prefill_seq_len=0,max_decode_seq_len=kv+1,query_start_loc=None,seq_start_loc=None,context_lens_tensor=None,block_tables=tables,use_cuda_graph=True)
     x=torch.randn((physical,config.hidden_size),device='cuda');res=x.clone();pos=torch.full((physical,),kv,device='cuda',dtype=torch.int64);pos[batch:]=0
     def forward():
      hidden=x;residual=res
      for layer,kv_cache in zip(layers[:count],cache):hidden,residual=layer(pos,hidden,kv_cache,metadata,residual)
      return hidden,residual
     from contextlib import nullcontext
     from vllm.distributed import graph_capture
     capture_context=graph_capture() if a.include_tp_collectives else nullcontext()
     with capture_context as context:
      stream=context.stream if context is not None else torch.cuda.Stream()
      stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
       for _ in range(3):forward()
      torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph,stream=stream):out=forward()
     for _ in range(a.warmup):graph.replay()
     torch.cuda.synchronize()
     pairs=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(a.repetitions)]
     for start,end in pairs:start.record();end.record()
     torch.cuda.synchronize()
     for start,end in pairs:start.record();graph.replay();end.record()
     torch.cuda.synchronize();times=[start.elapsed_time(end) for start,end in pairs]
     rank_times=[None]*tp
     torch.distributed.all_gather_object(rank_times,times)
     # Captured collectives synchronize ranks; queued replays remove Python
     # dispatch skew from the steady-state GPU primitive. Retain every rank.
     if a.include_tp_collectives:times=[max(values) for values in zip(*rank_times)]
     row=dict(model_name=a.model_name_for_vidur or a.model,tp_collective_backend="nccl" if a.include_tp_collectives else "excluded",rank_node_map=json.dumps([0]*tp),network_transport="local",includes_tp_collectives=a.include_tp_collectives,rank_gpu_samples_json=json.dumps(rank_times),model_path=a.model,num_tensor_parallel_workers=tp,num_layers=count,batch_size=batch,physical_batch_size=physical,kv_cache_size=kv,decode_block_table_capacity=a.capacity,execution_mode='cuda_graph',dtype='float16',phase='decode',block_size=16,n_embd=config.hidden_size,n_q_head=config.num_attention_heads,n_kv_head=config.num_key_value_heads,n_expanded_embd=config.intermediate_size,device_name=torch.cuda.get_device_name(),timing_semantics=('GPU span of native decoder DAG including native TP collectives; excludes embedding, final norm, logits, PP protocol, and driver' if a.include_tp_collectives else 'GPU span of native decoder DAG; excludes embedding, final norm, logits, TP collectives, PP protocol, and driver'))
     put_stats(row,'decoder_graph',times);rows.append(row)
     if rank==0:print(json.dumps(row),flush=True)
     del graph,out,pairs,metadata,x,res,pos,tables,slots;gc.collect();torch.cuda.empty_cache()
    del cache;gc.collect();torch.cuda.empty_cache()
 if rank==0:write_csv(a.part_output,rows)
finally:close_tp()
