"""Collect primitives through an isolated native streaming HTTP service."""
import csv,json,os,signal,socket,subprocess,time
from pathlib import Path
from http_profile_arguments import server_arguments,client_arguments


def run(args):
    from vllm_051_cpu_overhead import _write_rows
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    folder=output.parent/(output.stem+f'.http_profile.{os.getpid()}')
    folder.mkdir(exist_ok=False)
    driver=folder/'driver'
    with socket.socket() as test:
        test.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        test.bind(('127.0.0.1',8089))
    source=Path(__file__).resolve().parent
    environment=os.environ.copy()
    environment['PYTHONPATH']=str(source)+os.pathsep+environment.get('PYTHONPATH','')
    environment['NCCL_DEBUG']='INFO'
    command=server_arguments(args,source/'vllm_051_http_profile_server.py',driver)
    (folder/'server_launch.json').write_text(json.dumps(command,indent=2))
    log=(folder/'server.log').open('w');server=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
    try:
        import urllib.request
        for _ in range(180):
            if server.poll() is not None:raise RuntimeError(f'HTTP profile server failed; see {folder}/server.log')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8089/health',timeout=1):break
            except Exception:time.sleep(1)
        else:raise RuntimeError('HTTP profile server did not become healthy')
        measured=folder/'primitives.csv'
        command=client_arguments(args,source/'vllm_051_http_profile_client.py',driver,measured)
        (folder/'client_launch.json').write_text(json.dumps(command,indent=2))
        with (folder/'client.log').open('w') as stream:
            subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
        rows=list(csv.DictReader(measured.open()))
        if any(int(r['num_nodes'])>1 for r in rows):
            if not args.network_transport:raise ValueError('Cross-node profiling needs a transport verified in the NCCL log')
            raw=(folder/'server.log').read_text()
            if args.network_transport=='ib' and 'Using network Socket' in raw:
                raise ValueError('Requested IB profile but NCCL selected Socket; see server.log')
        _write_rows(output,rows,args.append)
        output.with_suffix('.metadata.json').write_text(json.dumps(dict(
            source_profile=str(measured),measurement='native HTTP serving primitives; no request E2E prediction target',
            arguments=vars(args)),indent=2))
        print('HTTP_CPU_PROFILE',output,flush=True)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill();server.wait(timeout=10)
        log.close()
