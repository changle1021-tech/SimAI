import pathlib,subprocess,time,os,signal,json,traceback,urllib.request
root=pathlib.Path(__file__).resolve().parent
for _ in range(1800):
    if (root/'phase_profile_v1.done').exists():break
    time.sleep(1)
else:raise RuntimeError('profile wait timeout')
for variant in ['shared_on','shared_off']:
    check=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for l in check.splitlines():
        i,m=map(int,l.split(','))
        if i in [4,5,6,7] and m>100:raise RuntimeError('GPU busy '+l)
    env=dict(os.environ,ALIGN_VARIANT=variant,VLLM_DISABLE_SHARED_EXPERTS_STREAM='0' if variant=='shared_on' else '1')
    out=root/'route_ablation'/variant;out.mkdir(parents=True,exist_ok=True)
    f=(out/'server.log').open('w');server=subprocess.Popen(['bash',str(root/'start_route_server.sh')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(out/'server.pid').write_text(str(server.pid));status={}
    try:
        for _ in range(900):
            if server.poll() is not None:raise RuntimeError('server exited '+str(server.returncode))
            try:urllib.request.urlopen('http://127.0.0.1:18180/health',timeout=2);break
            except OSError:time.sleep(1)
        else:raise RuntimeError('server readiness timeout')
        with (out/'collect.log').open('w') as log:p=subprocess.run([str(root/'env/bin/python'),'-u',str(root/'collect_route_ablation.py')],env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        status.update(returncode=p.returncode,success=p.returncode==0)
    except Exception:status['error']=traceback.format_exc()
    finally:
        if server.poll() is None:os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=60)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait(timeout=10)
        (out/'status.json').write_text(json.dumps(status,indent=2));print(variant,json.dumps(status),flush=True)
    if not status.get('success'):raise RuntimeError('ablation failed')
(root/'route_ablation.done').write_text('0')
