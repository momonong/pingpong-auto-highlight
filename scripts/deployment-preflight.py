#!/usr/bin/env python3
"""Read-only cutover guard. Requires an existing verified restore and explicit target data.

Refuses known writers in ANY Docker context. Does not stop or start services.
Host-native writers still require operator inventory before final backup.
"""
import argparse
import json
import subprocess
from pathlib import Path


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve(strict=True)
    conflicts = []
    contexts = docker('context', 'ls', '--format', '{{.Name}}').splitlines()
    for context in contexts:
        ids = docker('--context', context, 'ps', '-q').split()
        if not ids:
            continue
        for container in json.loads(docker('--context', context, 'inspect', *ids)):
            for mount in container['Mounts']:
                if mount['Type'] != 'bind' or not mount['RW']:
                    continue
                source = Path(mount['Source']).resolve()
                if source == data or source in data.parents or data in source.parents:
                    conflicts.append({'context': context, 'container': container['Name'],
                                      'source': str(source)})
    report = {'target_data': str(data), 'docker_contexts_checked': contexts,
              'conflicting_writers': conflicts}
    print(json.dumps(report, indent=2))
    if conflicts:
        raise SystemExit('Refusing a second writer; keep the current service unchanged')
    if not (data / 'state.sqlite3').is_file():
        raise SystemExit('No state.sqlite3: refusing bootstrap into unverified production data')


if __name__ == '__main__':
    main()
