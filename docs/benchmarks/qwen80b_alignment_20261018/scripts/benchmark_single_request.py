"""Serial, streaming single-request latency collection for a local vLLM server."""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request


def request(url, model, prompt, output_tokens, seed):
    payload = dict(model=model, prompt=prompt, max_tokens=output_tokens,
                   temperature=0, ignore_eos=True, seed=seed, stream=True,
                   stream_options={"include_usage": True})
    req = urllib.request.Request(url.rstrip('/') + '/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter_ns()
    events, usage = [], None
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            if not raw.startswith(b'data:'):
                continue
            data = raw[5:].strip()
            if data == b'[DONE]':
                break
            item = json.loads(data)
            now = (time.perf_counter_ns() - start) / 1e6
            if item.get('error'):
                raise RuntimeError(item['error'])
            if item.get('usage'):
                usage = item['usage']
            for choice in item.get('choices', []):
                # The completion API has no chat-role prelude. Preserve all
                # chunks, including empty token text, for timing audits.
                if choice.get('finish_reason') is None or choice.get('text'):
                    events.append(dict(ms=now, text=choice.get('text', ''),
                                       finish_reason=choice.get('finish_reason')))
    finished_ms = (time.perf_counter_ns() - start) / 1e6
    if not usage or usage['prompt_tokens'] != len(prompt) or usage['completion_tokens'] != output_tokens:
        raise RuntimeError(f'Token count mismatch: {usage}, expected {len(prompt)}/{output_tokens}')
    if not events:
        raise RuntimeError('No token events received')
    ttft = events[0]['ms']
    return dict(input_tokens=len(prompt), output_tokens=output_tokens,
                ttft_ms=ttft, e2e_ms=finished_ms,
                tpot_ms=(finished_ms-ttft)/(output_tokens-1) if output_tokens > 1 else None,
                last_token_event_ms=events[-1]['ms'], stream_events=events,
                usage=usage)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:18180')
    p.add_argument('--model', default='qwen80b-audit')
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--inputs', default='512,2048,8192')
    p.add_argument('--outputs', default='1,128')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    text = ('性能分析需要固定模型精度、并行方式和请求长度。'
            'A latency benchmark compares prefill, decode, communication, and host scheduling. '
            'For each request, preserve the exact input token IDs and output token count.\n')
    inputs = [int(s) for s in args.inputs.split(',')]
    outputs = [int(s) for s in args.outputs.split(',')]
    tokens = tokenizer.encode(text * (max(inputs) // 20 + 20), add_special_tokens=False)
    assert len(tokens) >= max(inputs)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args), indent=2))
    summaries = []
    with (out/'requests.jsonl').open('w') as f:
        for n in inputs:
            prompt = tokens[:n]
            (out/f'prompt_{n}.json').write_text(json.dumps(prompt))
            for m in outputs:
                samples = []
                for i in range(args.warmup + args.repeats):
                    row = request(args.url, args.model, prompt, m, 42)
                    row.update(repetition=i-args.warmup, warmup=i < args.warmup)
                    f.write(json.dumps(row, ensure_ascii=False)+'\n')
                    f.flush()
                    print(json.dumps({k: v for k, v in row.items() if k not in ('stream_events','usage')}), flush=True)
                    if not row['warmup']:
                        samples.append(row)
                summary = dict(input_tokens=n, output_tokens=m, repetitions=len(samples))
                for metric in ('ttft_ms', 'tpot_ms', 'e2e_ms'):
                    values = [s[metric] for s in samples if s[metric] is not None]
                    summary[metric] = statistics.median(values) if values else None
                    summary[metric+'_min'] = min(values) if values else None
                    summary[metric+'_max'] = max(values) if values else None
                summaries.append(summary)
                (out/'summary.json').write_text(json.dumps(summaries, indent=2))
                print('SUMMARY '+json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
