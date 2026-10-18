import pathlib,subprocess,json,time,os,signal,traceback,urllib.request
root=pathlib.Path(__file__).resolve().parent
check=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
(root/'gpu_before_components.csv').write_text(check)
for line in check.splitlines():
    ix,mem=map(int,line.split(','))
    if ix in [4,5,6,7] and mem>100:raise RuntimeError('GPU busy: '+line)
f=(root/'component_server.log').open('w');p=subprocess.Popen(['bash',str(root/'start_server.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(root/'component_server.pid').write_text(str(p.pid));status={}
try:
    for _ in range(900):
        if p.poll() is not None:raise RuntimeError('server exited '+str(p.returncode))
        try:
            urllib.request.urlopen('http://127.0.0.1:18180/health',timeout=2);break
        except OSError:time.sleep(1)
    else:raise RuntimeError('server not ready')
    with (root/'component_collect.log').open('w') as out:
        q=subprocess.run([str(root/'env/bin/python'),'-u',str(root/'collect_components.py')],stdout=out,stderr=subprocess.STDOUT,cwd=root,timeout=1200)
    status['collector_returncode']=q.returncode;status['success']=q.returncode==0
except Exception:status['error']=traceback.format_exc()
finally:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
    (root/'component_status.json').write_text(json.dumps(status,indent=2));(root/'component_driver.done').write_text('0' if status.get('success') else '1')
    (root/'gpu_after_components.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv'],text=True))
print(json.dumps(status),flush=True)
