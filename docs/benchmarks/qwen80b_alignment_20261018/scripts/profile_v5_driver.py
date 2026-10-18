import pathlib,time,json,subprocess,os,traceback
r=pathlib.Path(__file__).resolve().parent
while not (r/'native_chain.done').exists():time.sleep(1)
try:
 status=json.loads((r/'native_chain.done').read_text());assert status.get('success'),status
 env=dict(os.environ,PYTHONPATH=str(r/'source/SimAI/aicb'))
 subprocess.run([str(r/'env/bin/python'),str(r/'build_continuous_moe_profile.py')],env=env,check=True)
 subprocess.run(['python3','-u',str(r/'phase_profile_v5.py')],check=True)
except Exception:
 traceback.print_exc();(r/'phase_profile_v5.done').write_text('1')
