import pathlib,subprocess,time,os,signal,json,traceback,urllib.request
root=pathlib.Path(__file__).resolve().parent
while not (root/'continuous_moe.done').exists():time.sleep(1)
out=root/'native_chain';out.mkdir(exist_ok=True)
f=(out/'server.log').open('w');server=subprocess.Popen(['bash',str(root/'start_native_chain_server.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(out/'server.pid').write_text(str(server.pid));status={}
try:
 for _ in range(900):
  if server.poll() is not None:raise RuntimeError('server exited '+str(server.returncode))
  try:urllib.request.urlopen('http://127.0.0.1:18180/health',timeout=2);break
  except OSError:time.sleep(1)
 else:raise RuntimeError('server readiness timeout')
 with (out/'collect.log').open('w') as log:p=subprocess.run([str(root/'env/bin/python'),'-u',str(root/'collect_native_chain.py')],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 status.update(success=p.returncode==0,returncode=p.returncode)
except Exception:status['error']=traceback.format_exc()
finally:
 if server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
 try:server.wait(timeout=60)
 except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
 (out/'status.json').write_text(json.dumps(status,indent=2));(root/'native_chain.done').write_text(json.dumps(status));print(json.dumps(status),flush=True)
