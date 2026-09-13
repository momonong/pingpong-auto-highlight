#!/usr/bin/env python3
"""Package only reviewed deployment files and a locally inspected image (no data/secrets)."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(*args):
    return subprocess.check_output(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--context', default='default')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if run('git', '-C', str(root), 'status', '--porcelain').strip():
        raise SystemExit('Commit reviewed changes first; bundle must reference a clean worktree')
    revision = run('git', '-C', str(root), 'rev-parse', 'HEAD').decode().strip()
    image = json.loads(run('docker', '--context', args.context, 'image', 'inspect', args.image))[0]
    output = args.output.resolve()
    output.mkdir(mode=0o755, parents=True, exist_ok=False)
    names = ['compose.production.yaml', 'docs/deployment.md', 'docs/evaluation.md',
             'docs/architecture.md', 'scripts/deployment-data.py',
             'scripts/deployment-preflight.py', 'scripts/package-deployment.py',
             'tests/browser/proxy-deployment.cjs', 'tests/browser/proxy-real-playback.cjs',
             'tests/browser/proxy-upload-limits.cjs', 'tests/browser/proxy-ui-upload.cjs']
    names += run('git', '-C', str(root), 'ls-files', 'deploy',
                 'docs/evidence/deployment-integration-20260913').decode().splitlines()
    hashes = {}
    for name in names:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = run('git', '-C', str(root), 'show', revision + ':' + name)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    subprocess.run(['docker', '--context', args.context, 'image', 'save',
                    '-o', str(output / 'image.tar'), image['Id']], check=True)
    with (output / 'image.tar').open('rb') as f:
        hashes['image.tar'] = hashlib.file_digest(f, 'sha256').hexdigest()
    receipt = {'kind': 'LOCAL CANDIDATE - NOT A PUBLISHED RELEASE',
               'deployment_commit': revision,
               'runtime_commit': image['Config']['Labels']['org.opencontainers.image.revision'],
               'image_id': image['Id'], 'registry_digest': None,
               'docker_context': args.context, 'files': hashes}
    (output / 'bundle.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (output / 'SHA256SUMS').write_text(''.join(f'{sha}  {p}\n' for p, sha in hashes.items()))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
