"""Publish two recoverable snapshots using isolated Git indexes.

Never changes a working-tree file, HEAD, ordinary index or upstream remote.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

S = Path('/home/turbo_ops/changle/SimAI')
R = Path('/home/turbo_ops/changle/Files/analysis/pp_rollback_sync_20261002')
P = S / 'vidur-alibabacloud'
C = Path('/home/turbo_ops/changle/Files/vidur_cpu_overhead')
BRANCH = 'codex/pp-rollback-20261002'
OWNER = 'changle1021-tech'
PHASE = sys.argv[1]
assert PHASE in ('before', 'after')
E = os.environ.copy()
E.update(GIT_TERMINAL_PROMPT='0', GIT_SSH_COMMAND='ssh -o BatchMode=yes')
STATE = R / (PHASE + '_github_state.json')
state = {'phase': PHASE, 'status': 'preparing', 'started': time.time(),
         'branch': BRANCH, 'published': [], 'missing_forks_policy': 'archive_changes_and_versions_in_SimAI'}


def save():
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(STATE)


def run(p, args, env=None, data=None):
    out = subprocess.run(['git', '-C', str(p), *args], env=env or E,
                         input=data, capture_output=True, timeout=300)
    if out.returncode:
        raise RuntimeError('git ' + ' '.join(args[:2]) + ': ' + out.stderr.decode(errors='replace'))
    return out.stdout.decode().strip()


def add_blob(p, env, path, data, mode='100644'):
    oid = run(p, ['hash-object', '-w', '--stdin'], env, data)
    run(p, ['update-index', '--add', '--cacheinfo', mode, oid, path], env)


def snapshot(rel, name, overrides=None, modules=None, extra=None):
    p = S / rel
    head = run(p, ['rev-parse', 'HEAD'])
    base = head
    if PHASE == 'after':
        before = json.loads((R / 'before_github_state.json').read_text())
        base = next(x['commit'] for x in before['published'] if x['name'] == name)
    index = R / (PHASE + '_' + name + '.index')
    if index.exists():
        index.unlink()
    env = E.copy()
    env['GIT_INDEX_FILE'] = str(index)
    run(p, ['read-tree', head], env)
    run(p, ['add', '-A'], env)
    for path, oid in (overrides or {}).items():
        run(p, ['update-index', '--add', '--cacheinfo', '160000', oid, path], env)
    if modules is not None:
        add_blob(p, env, '.gitmodules', modules.encode())
    for path, data in (extra or {}).items():
        add_blob(p, env, path, data)
    tree = run(p, ['write-tree'], env)
    message = ('Archive all current local changes before PP rollback' if PHASE == 'before'
               else 'Restore original TP implementation and archive state after PP rollback')
    commit = run(p, ['commit-tree', tree, '-p', base], env,
                 (message + '\n\nRepository: ' + name + '\nSnapshot date: 2026-10-02\n').encode())
    ref = 'refs/heads/' + BRANCH
    existing = subprocess.run(['git', '-C', str(p), 'rev-parse', '--verify', ref],
                              env=E, capture_output=True)
    old = existing.stdout.decode().strip() if existing.returncode == 0 else '0' * 40
    run(p, ['update-ref', ref, commit, old])
    return {'name': name, 'path': str(p), 'original_head': head,
            'parent': base, 'commit': commit, 'tree': tree,
            'url': 'git@github.com:' + OWNER + '/' + name + '.git'}


def publish(item):
    p = Path(item['path'])
    ref = 'refs/heads/' + BRANCH
    run(p, ['push', item['url'], item['commit'] + ':' + ref])
    observed = run(p, ['ls-remote', item['url'], ref]).split()
    if not observed or observed[0] != item['commit']:
        raise RuntimeError('Remote ref verification failed for ' + item['name'])
    item['verified_remote_commit'] = observed[0]
    state['published'].append(item)
    save()
    print(json.dumps({'published': item['name'], 'phase': PHASE, 'commit': item['commit']}), flush=True)


try:
    save()
    if PHASE == 'after':
        rollback = json.loads((R / 'rollback_state.json').read_text())
        verification = json.loads((R / 'validation_state.json').read_text())
        assert rollback['status'] == 'complete'
        assert verification['status'] == 'complete'
    archive = R / 'snapshot_payload'
    archive.mkdir(exist_ok=True)
    nested = [
        ('DeepGEMM', 'DeepGEMM'),
        ('DeepGEMM/third-party/cutlass', 'cutlass'),
        ('DeepGEMM/third-party/fmt', 'fmt'),
        ('aicb/src/sarathi', 'sarathi-serve'),
    ]
    nested_records = []
    for rel, name in nested:
        p = S / rel
        head = run(p, ['rev-parse', 'HEAD'])
        record = {'name': name, 'path': rel, 'head': head,
                  'status': run(p, ['status', '--short']),
                  'source_url': run(p, ['remote', 'get-url', 'origin'])}
        if PHASE == 'before' and name == 'DeepGEMM':
            item = snapshot(rel, name)
            bundle = archive / 'DeepGEMM-local-changes.bundle'
            run(p, ['bundle', 'create', str(bundle), 'refs/heads/' + BRANCH, '^' + head])
            run(p, ['bundle', 'verify', str(bundle)])
            record['archived_commit'] = item['commit']
            record['bundle_sha256'] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        nested_records.append(record)
    (archive / ('nested-repositories-' + PHASE + '.json')).write_text(json.dumps(nested_records, indent=2))
    external = archive / ('external-' + PHASE)
    external.mkdir(exist_ok=True)
    for q in C.rglob('*'):
        if q.is_file() and '__pycache__' not in q.parts and q.suffix in ('.py', '.csv'):
            target = external / q.relative_to(C)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(q.read_bytes())
    child_commits = {}
    sarathi_head = run(S / 'aicb/src/sarathi', ['rev-parse', 'HEAD'])
    aicb_modules = ('[submodule "src/sarathi"]\n\tpath = src/sarathi\n'
                    '\turl = https://github.com/microsoft/sarathi-serve.git\n')
    for rel, name in [('aicb', 'aicb'), ('SimCCL', 'SimCCL'),
                      ('ns-3-alibabacloud', 'ns-3-alibabacloud')]:
        item = snapshot(rel, name,
                        overrides={'src/sarathi': sarathi_head} if name == 'aicb' else None,
                        modules=aicb_modules if name == 'aicb' else None)
        publish(item)
        child_commits[rel] = item['commit']
    modules = ''.join('[submodule "' + rel + '"]\n\tpath = ' + rel + '\n'
                      '\turl = https://github.com/' + OWNER + '/' + name + '.git\n'
                      for rel, name in [('SimCCL', 'SimCCL'), ('aicb', 'aicb'),
                                        ('ns-3-alibabacloud', 'ns-3-alibabacloud')])
    modules += ('[submodule "DeepGEMM"]\n\tpath = DeepGEMM\n'
                '\turl = https://github.com/deepseek-ai/DeepGEMM.git\n')
    prefix = 'docs/pp-rollback-20261002/'
    extra = {prefix + str(q.relative_to(archive)): q.read_bytes()
             for q in archive.rglob('*') if q.is_file()}
    extra[prefix + 'pre-rollback-evidence.zip'] = (R / 'pp_pre_rollback_evidence_20261002.zip').read_bytes()
    for filename in ['inventory.json', 'rollback_plan.json', 'rollback_state.json', 'validation_state.json',
                     'preserved_controls.json', 'pp_github_snapshot_remote.py', 'pp_selective_rollback_remote.py']:
        q = R / filename
        if q.is_file():
            extra[prefix + filename] = q.read_bytes()
    if PHASE == 'after':
        for name in ['restored_tp_tests.log', 'restored_tp2.launch.json', 'restored_tp4.launch.json',
                     'restored_tp2.log', 'restored_tp4.log']:
            q = R / name
            if q.is_file():
                extra[prefix + 'validation/' + name] = q.read_bytes()
        for case in ['restored_tp2', 'restored_tp4']:
            for q in (R / case).rglob('*'):
                if q.is_file() and q.name in ['request_metrics.csv', 'batch_metrics.csv', 'config.json']:
                    extra[prefix + 'validation/' + str(q.relative_to(R))] = q.read_bytes()
    extra[prefix + 'README.md'] = ('''# PP rollback archive\n\nTwo snapshots are published on `codex/pp-rollback-20261002`.\n\nThe before snapshot retains the PP implementation, native profiles, experiment records, original TP backups and external CPU entrypoints/tables. The after snapshot restores the original TP source and CPU table. No main branch or local working index is changed by publication. The three existing child forks are published first, and this snapshot points at their verified commits.\n\nDeepGEMM, cutlass, fmt and sarathi-serve forks were unavailable. Their exact original versions and source URLs are recorded in `nested-repositories-before.json`. DeepGEMM local changes are preserved in the verified Git bundle, whose prerequisite is the recorded original DeepGEMM commit. The other three have no local changes. Upstream URLs remain for those clean dependencies.\n\nTo recover DeepGEMM changes, clone the recorded upstream repository, ensure the prerequisite commit is present, and fetch `DeepGEMM-local-changes.bundle` with `refs/heads/codex/pp-rollback-20261002` as the source ref. No model weights or transient caches are included.\n\nHistorical reports inside the ZIP describe intermediate PP experiments; see the original-path regression report for the later TP regression findings.\n''').encode()
    item = snapshot('', 'SimAI', overrides=child_commits, modules=modules, extra=extra)
    publish(item)
    state.update(status='complete', completed=time.time())
    save()
except Exception as exc:
    state.update(status='failed', error=repr(exc), completed=time.time())
    save()
    raise
