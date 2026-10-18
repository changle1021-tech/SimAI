"""Collect separate graph traces or eager CUDA event measurements."""
import argparse
import json
import pathlib
import time
import urllib.request

from benchmark_single_request import request

ROOT = pathlib.Path('/home/turbo_ops/changle/Files/analysis/qwen80b_alignment_20261016')
URL = 'http://127.0.0.1:18180'


def rpc(method, *args):
    req = urllib.request.Request(URL+'/collective_rpc',
                                 data=json.dumps(dict(method=method, args=list(args))).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['graph', 'eager'])
    args = p.parse_args()
    out = ROOT/(args.mode+'_audit')
    out.mkdir(parents=True, exist_ok=True)
    for _ in range(600):
        try:
            with urllib.request.urlopen(URL+'/health', timeout=2):
                break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError('Server did not become ready')
    info = rpc('q80_info')
    (out/'workers.json').write_text(json.dumps(info, indent=2))
    print('WORKERS '+json.dumps(info), flush=True)
    records = []

    def run(n, m, label):
        prompt = json.loads((ROOT/'baseline'/f'prompt_{n}.json').read_text())
        row = request(URL, 'qwen80b-audit', prompt, m, 42)
        row.update(label=label, mode=args.mode)
        records.append(row)
        (out/'requests.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(json.dumps({k:v for k,v in row.items() if k not in ('stream_events', 'usage')}), flush=True)

    for n in (512, 2048):
        for i in range(5):
            run(n, 16, 'warmup')
        for i in range(5):
            run(n, 16, 'uninstrumented_16')
        for i in range(3):
            run(n, 256, 'uninstrumented_256')
        if args.mode == 'graph':
            for m in (1, 16):
                label = f'graph_p{n}_o{m}'
                print('TRACE_START '+label, flush=True)
                rpc('q80_start_trace', label)
                try:
                    run(n, m, 'torch_profiler')
                finally:
                    result = rpc('q80_stop_trace')
                    (out/(label+'_paths.json')).write_text(json.dumps(result, indent=2))
                print('TRACE_STOP '+label, flush=True)
        else:
            for m in (1, 16):
                label = f'eager_events_p{n}_o{m}'
                print('EVENT_START '+label, flush=True)
                rpc('q80_start_events')
                try:
                    run(n, m, 'cuda_events')
                finally:
                    result = rpc('q80_stop_events', label)
                    (out/(label+'_paths.json')).write_text(json.dumps(result, indent=2))
                print('EVENT_STOP '+label, flush=True)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
