import pathlib,time,os,json,statistics as st,traceback,types,subprocess
root=pathlib.Path(__file__).resolve().parent
for _ in range(2400):
    if (root/'route_ablation.done').exists():break
    time.sleep(1)
else:raise RuntimeError('route ablation wait timeout')
os.environ['CUDA_VISIBLE_DEVICES']='4';os.environ['OMP_NUM_THREADS']='1'
import torch
from workload_generator.mocked_model.inference import qwen3_next_projections as p
from workload_generator.mocked_model.inference import qwen3_next_attention as a
from workload_generator.mocked_model.inference import qwen3_next_moe as m
args=types.SimpleNamespace(**json.loads((root/'source/SimAI/aicb/scripts/inference_configs/qwen3_next_default.json').read_text()))
args.phase='decode';args.tensor_model_parallel_size=1;args.world_size=4;args.pipeline_model_parallel=1;args.expert_model_parallel_size=4
torch.manual_seed(42)
flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)

def amortized(fn,cold,repeats=64,samples=20):
    for _ in range(5):fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            if cold:flush.zero_()
            fn()
    baseline=None
    if cold:
        baseline=torch.cuda.CUDAGraph()
        with torch.cuda.graph(baseline):
            for _ in range(repeats):flush.zero_()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    def elapsed(g):
        start.record();g.replay();end.record();end.synchronize();return start.elapsed_time(end)*1000/repeats
    values=[];base=[]
    for _ in range(samples):
        x=elapsed(graph);b=elapsed(baseline) if baseline is not None else 0
        values.append(x-b);base.append(b)
    return dict(median_us=st.median(values),samples_us=values,baseline_us=st.median(base))

def capture_helper(module,func,*aa):
    box=[];original=module.benchmark_component;module.benchmark_component=lambda args,fn:box.append(fn) or 0
    try:func(*aa)
    finally:module.benchmark_component=original
    assert len(box)==1;return box[0]

x=torch.randn(1,2048,device='cuda',dtype=torch.bfloat16);w=torch.randn(12288,2048,device='cuda',dtype=torch.bfloat16);out=torch.empty_like(x)
cases=[('empty',lambda:None),('add_2048',lambda:torch.add(x,x,out=out)),('gdn_qkvz',lambda:torch.nn.functional.linear(x,w)),('conv_decode',capture_helper(a,a.benchmark_gdn_conv,args,'decode',1)),('gdn_core_decode',capture_helper(a,a.benchmark_gdn_core,args,'decode',1,512)),('shared_expert',capture_helper(m,m.benchmark_shared_expert,args,1)),('routed_experts',capture_helper(m,m.benchmark_routed_experts,args,1))]
rows=[]
for name,fn in cases:
    row={'component':name}
    try:
        row['external_events_us']=[p.benchmark_gpu_region(fn)*1e6 for _ in range(3)]
        row['amortized_warm']=amortized(fn,False)
        row['amortized_cold']=amortized(fn,True)
        # Kernel-only check on the same cold callback. The cache flush is a
        # recognizable fill kernel; retain all names and counts for auditing.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(12):
                flush.zero_();torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
        groups={}
        for ev in prof.events():
            if str(ev.device_type).endswith('CUDA'):
                item=groups.setdefault(ev.name,{'count':0,'total_us':0});item['count']+=1;item['total_us']+=ev.device_time_total
        row['profiled_kernels']=groups
    except Exception:row['error']=traceback.format_exc()
    rows.append(row);(root/'timer_method_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['profiled_kernels','amortized_cold','amortized_warm']}),flush=True)
(root/'timer_method_probe.done').write_text('0');print('COMPLETE',flush=True)
