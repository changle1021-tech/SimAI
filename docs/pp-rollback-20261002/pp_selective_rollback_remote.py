"""Restore explicitly owned PP edits only after all before snapshots are verified."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

R = Path('/home/turbo_ops/changle/Files/analysis/pp_rollback_sync_20261002')
P = Path('/home/turbo_ops/changle/SimAI/vidur-alibabacloud')
C = Path('/home/turbo_ops/changle/Files/vidur_cpu_overhead')
PY = '/home/turbo_ops/miniconda3/envs/sarathi/bin/python'
CONTAINER = 'vllm051'
CONTAINER_CPU = '/vllm-workspace/vidur_cpu_overhead'
plan = json.loads((R / 'rollback_plan.json').read_text())
state = {'status': 'checking_before_snapshots', 'started': time.time(),
         'restored': [], 'removed': [], 'container': {}}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save():
    (R / 'rollback_state.json').write_text(json.dumps(state, indent=2))


def command(args, **kwargs):
    out = subprocess.run(args, capture_output=True, text=True, timeout=300, **kwargs)
    if out.returncode:
        raise RuntimeError('Command failed: ' + repr(args[:5]) + '\n' + out.stderr[-4000:])
    return out.stdout


def check_controls():
    values = json.loads((R / 'preserved_controls.json').read_text())
    for path, expected in values.items():
        assert sha(path) == expected, 'Unrelated TP source/input changed: ' + path
    return len(values)


try:
    save()
    before = json.loads((R / 'before_github_state.json').read_text())
    assert before['status'] == 'complete'
    assert {x['name'] for x in before['published']} == {'SimAI', 'aicb', 'SimCCL', 'ns-3-alibabacloud'}
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT='0', GIT_SSH_COMMAND='ssh -o BatchMode=yes')
    for item in before['published']:
        remote = command(['git', 'ls-remote', item['url'], 'refs/heads/' + before['branch']], env=env).split()
        assert remote and remote[0] == item['commit'], 'Before snapshot no longer reachable: ' + item['name']
    state['preserved_controls_before'] = check_controls()
    for item in plan['restore'] + plan['remove']:
        assert sha(item['path']) == item['before_sha256'], 'File changed after inspection: ' + item['path']
    for item in plan['restore']:
        assert sha(item['source']) == item['restore_sha256'], 'Original backup changed: ' + item['source']
    docker_prefix = ['sudo', '-n', 'docker', 'exec', CONTAINER, 'python3', '-c']
    read_container = (
        "from pathlib import Path;import hashlib,json;p=Path(" + repr(CONTAINER_CPU) + ");"
        "print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() "
        "for f in p.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix in ['.py','.csv']}))"
    )
    container_before = json.loads(command(docker_prefix + [read_container]))
    container_owned = {}
    for item in plan['restore'] + plan['remove']:
        if item['relative_path'].startswith('external/'):
            rel = item['relative_path'][len('external/'):]
            if rel in container_before:
                assert container_before[rel] == item['before_sha256'], 'Container file differs from archived host copy: ' + rel
                container_owned[rel] = item
    state['container']['before_sha256'] = container_before
    # Container copies were checked against the external-before files in GitHub.
    state['status'] = 'restoring'
    save()
    saved = R / 'removed_and_replaced'
    for item in plan['restore'] + plan['remove']:
        target = Path(item['path'])
        backup = saved / item['relative_path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        assert sha(backup) == item['before_sha256']
        if item in plan['restore']:
            if item.get('sudo'):
                command(['sudo', '-n', 'cp', item['source'], str(target)])
            else:
                temp = target.with_name(target.name + '.pp-rollback.tmp')
                shutil.copy2(item['source'], temp)
                temp.replace(target)
            assert sha(target) == item['restore_sha256']
            state['restored'].append({'path': str(target), 'sha256': sha(target)})
        else:
            if item.get('sudo'):
                command(['sudo', '-n', 'rm', '--', str(target)])
            else:
                target.unlink()
            assert not target.exists()
            state['removed'].append(str(target))
        save()
    # Restore the existing user's container entrypoint; do not restart any service.
    command(['sudo', '-n', 'docker', 'cp', str(C / 'vllm_051_cpu_overhead.py'),
             CONTAINER + ':' + CONTAINER_CPU + '/vllm_051_cpu_overhead.py'])
    container_remove = [rel for rel, item in container_owned.items() if item in plan['remove']]
    remove_code = (
        'from pathlib import Path;import json,hashlib;p=Path(' + repr(CONTAINER_CPU) + ');'
        'files=json.loads(' + repr(json.dumps({rel: container_before[rel] for rel in container_remove})) + ');'
        '\nfor rel,expected in files.items():\n q=p/rel\n assert hashlib.sha256(q.read_bytes()).hexdigest()==expected\n q.unlink()\n'
    )
    command(docker_prefix + [remove_code])
    container_after = json.loads(command(docker_prefix + [read_container]))
    original_profiler = next(x['restore_sha256'] for x in plan['restore']
                             if x['relative_path'] == 'external/vllm_051_cpu_overhead.py')
    assert container_after['vllm_051_cpu_overhead.py'] == original_profiler
    assert not set(container_remove).intersection(container_after)
    state['container']['after_sha256'] = container_after
    state['preserved_controls_after'] = check_controls()
    cpu_rows = list(csv.DictReader((C / 'cpu_overheads.csv').open()))
    assert len(cpu_rows) == 24
    assert {int(x['tensor_parallel_degree']) for x in cpu_rows} == {1, 2, 4}
    state['restored_cpu_rows'] = len(cpu_rows)
    for item in plan['restore']:
        p = Path(item['path'])
        if p.suffix == '.py':
            ast.parse(p.read_text(), filename=str(p))
    state.update(status='complete', completed=time.time())
    save()
except Exception as exc:
    state.update(status='failed', error=repr(exc), completed=time.time())
    save()
    raise

validation = {'status': 'running', 'started': time.time(), 'tests': {}, 'cases': []}


def save_validation():
    (R / 'validation_state.json').write_text(json.dumps(validation, indent=2))


try:
    save_validation()
    environment = os.environ.copy()
    environment.update(WANDB_MODE='disabled', PYTHONPATH=str(P), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', LOKY_MAX_CPU_COUNT='4')
    with (R / 'restored_tp_tests.log').open('w') as log:
        test = subprocess.run([PY, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
                              cwd=P, env=environment, stdout=log, stderr=subprocess.STDOUT)
    validation['tests'] = {'exit_code': test.returncode, 'log': str(R / 'restored_tp_tests.log')}
    assert test.returncode == 0, 'Existing TP tests failed; see restored_tp_tests.log'
    save_validation()
    legacy = Path('/home/turbo_ops/changle/Files/analysis/tp_legacy_regression_20261002')
    for name, tp, baseline in [('A_original_runtime_original_inputs', 4, 376.02657537208916),
                               ('G_original_tp2', 2, 371.6817069129864)]:
        launch = json.loads((legacy / (name + '.launch.json')).read_text())
        args = launch['cmd'][:]
        out = R / ('restored_tp' + str(tp))
        i = args.index('--metrics_config_output_dir')
        args[i + 1] = str(out)
        # Same original simulation settings and warm cache; only cwd/output change.
        launch.update(cmd=args, cwd=str(P), PYTHONPATH=str(P))
        (R / ('restored_tp' + str(tp) + '.launch.json')).write_text(json.dumps(launch, indent=2))
        with (R / ('restored_tp' + str(tp) + '.log')).open('w') as log:
            case = subprocess.run(args, cwd=P, env=environment, stdout=log, stderr=subprocess.STDOUT)
        assert case.returncode == 0, 'Restored TP simulation failed: TP' + str(tp)
        files = sorted(out.rglob('request_metrics.csv'))
        assert len(files) == 1
        rows = list(csv.DictReader(files[0].open()))
        assert len(rows) == 3
        e2e = [float(x['request_e2e_time']) * 1000 for x in rows]
        observed = sum(e2e) / len(e2e)
        result = {'tp': tp, 'pp': 1, 'requests': len(rows), 'e2e_ms': e2e,
                  'mean_e2e_ms': observed, 'original_path_baseline_ms': baseline,
                  'change_from_original_path_pct': (observed / baseline - 1) * 100,
                  'request_metrics': str(files[0]), 'launch_record': str(R / ('restored_tp' + str(tp) + '.launch.json')),
                  'scope': 'Restoration regression against original simulation path; not a new native accuracy measurement'}
        validation['cases'].append(result)
        save_validation()
        print(json.dumps(result), flush=True)
        # RF cache re-fitting can introduce tiny reproducibility variation. This checks restoration, not calibration.
        assert abs(observed / baseline - 1) < 0.01, 'Original TP simulation behavior did not reproduce'
    validation['unchanged_controls'] = check_controls()
    validation.update(status='complete', completed=time.time())
    save_validation()
except Exception as exc:
    validation.update(status='failed', error=repr(exc), completed=time.time())
    save_validation()
    raise
