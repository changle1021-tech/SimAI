import pathlib,json,subprocess,os,csv,time
r=pathlib.Path(__file__).resolve().parent
while not (r/'phase_profile_v4.done').exists():time.sleep(1)
if (r/'phase_profile_v4.done').read_text()!='0':raise RuntimeError('Component grid generation failed')
base='''--no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace --replica_config_network_device h100_dgx --replica_config_device h100 --replica_config_nvlink_bandwidth 3600 --poisson_request_interval_generator_config_qps 512 --synthetic_request_generator_config_num_requests 1 --length_generator_config_type fixed --fixed_request_length_generator_config_max_tokens 32768 --interval_generator_config_type poisson --cluster_config_num_replicas 4 --replica_config_pd_node_ratio 1 --global_scheduler_config_type lor --replica_scheduler_config_type sarathi --sarathi_scheduler_config_chunk_size 512 --replica_config_model_name qwen3-next-80B --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 --random_forrest_execution_time_predictor_config_backend aicb --random_forrest_execution_time_predictor_config_prediction_max_batch_size 1 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 9216 --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 512'''.split()
base+=['--no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling','--random_forrest_execution_time_predictor_config_aicb_profile_dir',str(r/'phase_profile_v4/workload'),'--random_forrest_execution_time_predictor_config_aicb_cpu_overhead_profile',str(r/'cpu_boundary_profile_v3.json'),'--random_forrest_execution_time_predictor_config_aicb_moe_comm_profile',str(r/'comm_profile_v2.json')]
rows=[]
for n,m in [(512,256),(2048,256),(8192,256),(256,64),(1024,128),(4096,128),(1536,192),(3072,192)]:
 out=r/'sim_alignment_v4_cpuv3'/f'p{n}_o{m}';out.mkdir(parents=True,exist_ok=True)
 cmd=['/home/turbo_ops/miniconda3/envs/vidur/bin/python',str(r/'run_vidur_alignment.py'),*base,'--fixed_request_length_generator_config_prefill_tokens',str(n),'--fixed_request_length_generator_config_decode_tokens',str(m),'--metrics_config_output_dir',str(out)]
 (out/'command.json').write_text(json.dumps(cmd,indent=2));env=dict(os.environ,QWEN_AUDIT_ROWS=str(out/'iterations.jsonl'),CUDA_VISIBLE_DEVICES='')
 with (out/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 row=dict(input_tokens=n,output_tokens=m,returncode=p.returncode)
 if p.returncode:row['error']=(out/'run.log').read_text()[-6000:]
 else:
  paths=list(out.rglob('request_metrics.csv'));assert len(paths)==1
  metrics=list(csv.DictReader(paths[0].open()));assert len(metrics)==1
  x=metrics[0];ttft=float(x['prefill_e2e_time'])*1e3;total=float(x['request_e2e_time'])*1e3
  row.update(ttft_ms=ttft,e2e_ms=total,tpot_ms=(total-ttft)/(m-1))
 rows.append(row);(r/'sim_alignment_v4_cpuv3_summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
(r/'sim_alignment_v4_cpuv3.done').write_text('0' if all(x['returncode']==0 for x in rows) else '1')
