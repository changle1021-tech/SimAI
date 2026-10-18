"""Observe the modeled component decomposition; no modeled costs are patched."""
import json,os,pathlib,runpy,sys
r=pathlib.Path(__file__).resolve().parent;repo=r/'source/SimAI/vidur-alibabacloud'
os.chdir(repo);sys.path.insert(0,str(repo))
from vidur.execution_time_predictor.base_execution_time_predictor import BaseExecutionTimePredictor
from vidur.entities.execution_time import ExecutionTime
out=pathlib.Path(os.environ['QWEN_AUDIT_ROWS']);out.parent.mkdir(parents=True,exist_ok=True)
original=BaseExecutionTimePredictor.get_execution_time

def observed(self,batch,pipeline_stage,*args,**kwargs):
 result=original(self,batch,pipeline_stage,*args,**kwargs);c=result._replica_config
 # model_time forces validated CSV loading; all subsequent fields share it.
 model_ms=result.model_time_ms
 rows=[]
 for i in range(c.model_config.num_layers):
  data=result._aicb_data[i]
  att=result._get_attention_layer_execution_time_from_aicb(i)*1e3
  moe=result._get_moe_layer_execution_time_from_aicb(i)*1e3
  comp=data['moe']['comp_time']*1e-6
  rows.append(dict(layer=i,attention_ms=att,moe_profile_ms=comp,moe_profile_includes_ep_comm=bool(data['moe'].get('includes_ep_comm',0)),additional_ep_comm_ms=moe-comp))
 row=dict(phase=c.phase,seq=c.seq_len,context=getattr(c,'context_len',0),batch_size=c.batch_size,model_ms=model_ms,cpu_boundary_ms=result._get_cpu_overhead(),total_ms=result.total_time*1e3,layers=rows)
 with out.open('a') as f:f.write(json.dumps(row)+'\n')
 return result
BaseExecutionTimePredictor.get_execution_time=observed
runpy.run_module('vidur.main',run_name='__main__')
