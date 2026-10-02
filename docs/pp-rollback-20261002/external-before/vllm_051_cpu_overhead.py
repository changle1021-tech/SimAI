from pathlib import Path
import runpy,sys
profile_dir=Path(__file__).resolve().parent/'single_node_pp_20261001'
sys.path.insert(0,str(profile_dir))
runpy.run_path(str(profile_dir/'vllm_051_cpu_overhead.py'),run_name='__main__')
