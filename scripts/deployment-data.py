#!/usr/bin/env python3
"""Stream a complete data snapshot; restore only into a NEW directory.

Run snapshot in the existing container so SQLite mode=ro can read its WAL/SHM.
Online snapshots are rehearsal material; final cutover requires stopped writers.
Never print file contents or credentials. The archive itself is sensitive.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def files(root, excluded):
    result = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root)
        if rel.parts[0] in excluded:
            continue
        if p.is_symlink():
            raise ValueError(f'Symlink requires explicit migration review: {rel}')
        if p.is_file() and not p.name.endswith(('-wal', '-shm', '-journal')):
            result.append(p)
    return result


def snapshot(args):
    root = args.source.resolve()
    excluded = set(args.exclude)
    paths = files(root, excluded)
    if not (root / 'state.sqlite3').is_file():
        raise ValueError('Missing state.sqlite3; refusing an empty deployment backup')
    initial = {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
               for p in paths}
    manifest = {'format': 1, 'created_utc': datetime.now(UTC).isoformat(),
                'source_root': str(root), 'consistency': args.consistency,
                'excluded_top_level': sorted(excluded), 'files': {}, 'databases': {}}
    connections = []
    with tempfile.TemporaryDirectory(prefix='hc-sqlite-backup-') as tmp:
        copies = {}
        try:
            for i, p in enumerate(paths):
                if p.suffix != '.sqlite3':
                    continue
                database_source = p
                if args.consistency == 'stopped-writers':
                    # Read-only mount may have no SHM after clean shutdown. Recover copied
                    # WAL/journal in scratch space, never create sidecars on the source.
                    raw = Path(tmp) / ('raw-' + str(i))
                    shutil.copyfile(p, raw)
                    for suffix in ('-wal', '-journal'):
                        sidecar = Path(str(p) + suffix)
                        if sidecar.exists():
                            shutil.copyfile(sidecar, Path(str(raw) + suffix))
                    database_source = raw
                mode = 'rw' if database_source != p else 'ro'
                c = sqlite3.connect(database_source.as_uri() + '?mode=' + mode,
                                    uri=True, timeout=30)
                version = c.execute('pragma data_version').fetchone()[0]
                connections.append((c, version))
                copy = Path(tmp) / str(i)
                with sqlite3.connect(copy) as target:
                    c.backup(target)
                    integrity = target.execute('pragma integrity_check').fetchone()[0]
                    if integrity != 'ok':
                        raise ValueError(f'Integrity failure: {p.name}')
                    fk = target.execute('pragma foreign_key_check').fetchall()
                    if fk:
                        raise ValueError(f'Foreign key failure: {p.name}')
                copies[p] = copy
                manifest['databases'][str(p.relative_to(root))] = {'integrity': integrity}
            with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
                for p in paths:
                    rel = str(p.relative_to(root))
                    actual = copies.get(p, p)
                    sha = digest(actual)
                    info = archive.gettarinfo(str(actual), arcname='data/' + rel)
                    info.mode = 0o600
                    info.uid = info.gid = 10001
                    info.uname = info.gname = ''
                    with actual.open('rb') as f:
                        archive.addfile(info, f)
                    # Detect in-place changes as well as renamed/replaced source files.
                    if digest(actual) != sha:
                        raise ValueError(f'File changed during backup: {rel}')
                    manifest['files'][rel] = {'size': actual.stat().st_size, 'sha256': sha}
                final = {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
                         for p in files(root, excluded)}
                if initial != final or any(c.execute('pragma data_version').fetchone()[0] != v
                                           for c, v in connections):
                    raise ValueError('Source changed during snapshot; retry with a new archive')
                manifest['observed_stable'] = True
                encoded = (json.dumps(manifest, indent=2) + '\n').encode()
                info = tarfile.TarInfo('manifest.json')
                info.mode = 0o600
                info.size = len(encoded)
                archive.addfile(info, io.BytesIO(encoded))
        finally:
            for c, _ in connections:
                c.close()


def inventory(root, source_root):
    databases = {}
    missing = []
    incompatible = []
    source_hashes = {}

    def mapped(raw):
        p = PurePosixPath(raw)
        try:
            rel = p.relative_to(source_root)
        except ValueError:
            incompatible.append(raw)
            return None
        if '..' in rel.parts:
            incompatible.append(raw)
            return None
        return root / str(rel)

    for p in sorted(root.rglob('*.sqlite3')):
        c = sqlite3.connect(p.as_uri() + '?mode=ro', uri=True)
        c.row_factory = sqlite3.Row
        tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
        report = {'integrity': c.execute('pragma integrity_check').fetchone()[0],
                  'foreign_keys': len(c.execute('pragma foreign_key_check').fetchall()),
                  'tables': {t: c.execute('select count(*) from "' + t + '"').fetchone()[0]
                             for t in tables}}
        if p.name == 'state.sqlite3':
            known = {'users', 'sessions', 'uploads', 'jobs', 'drive_imports',
                     'annotations', 'cleanup_queue'}
            if set(tables) - known:
                incompatible.append('unknown state tables:' + ','.join(sorted(set(tables) - known)))
        if 'uploads' in tables:
            report['active_jobs'] = c.execute(
                "select count(*) from jobs where status in ('queued','processing')").fetchone()[0]
            report['active_uploads'] = c.execute(
                "select count(*) from uploads where status != 'completed'").fetchone()[0]
            report['orphan_jobs'] = c.execute(
                'select count(*) from jobs j left join uploads u on j.upload_id=u.id '
                'where u.id is null').fetchone()[0]
            for row in c.execute('select path,size,offset,status from uploads'):
                target = mapped(row['path'])
                expected = row['size'] if row['status'] == 'completed' else row['offset']
                if target and (not target.is_file() or target.stat().st_size != expected):
                    missing.append(row['path'])
            for row in c.execute('select id,result_json from jobs where result_json is not null'):
                for f in json.loads(row['result_json']).get('files', []):
                    target = root / 'outputs' / row['id'] / f['name']
                    if not target.resolve().is_relative_to(root) or not target.is_file():
                        missing.append(str(target.relative_to(root)))
        if 'sources' in tables:
            report['reviews'] = []
            for row in c.execute('select id,payload from sources'):
                source = json.loads(row['payload'])
                target = mapped(source['path'])
                if target:
                    if target.is_file():
                        sha = source_hashes.setdefault(str(target), digest(target))
                        if sha != row['id']:
                            missing.append('source hash mismatch:' + row['id'])
                    else:
                        missing.append(source['path'])
            for row in c.execute('select source_id,payload from reviews'):
                review = json.loads(row['payload'])
                report['reviews'].append({'source_id': row['source_id'],
                                          'points': len(review.get('points', [])),
                                          'coverage': review.get('coverage', []),
                                          'payload_sha256': hashlib.sha256(
                                              row['payload'].encode()).hexdigest()})
        preserve_tables = {
            'highlight_clips', 'compilations', 'compilation_items', 'storage_objects',
        }
        if preserve_tables.intersection(tables):
            incompatible.append('preserve-local schema:' + str(p.relative_to(root)))
        databases[str(p.relative_to(root))] = report
        c.close()
    for p in root.rglob('preview.json'):
        spec = json.loads(p.read_text())
        # Keep unknown manifest shapes visible rather than silently rewriting paths.
        if 'path' in spec:
            target = mapped(spec['path'])
            if target and target.is_file() and target.stat().st_size != spec.get('size'):
                missing.append('preview size mismatch:' + str(p.relative_to(root)))
            if spec.get('source_id') != p.parent.name:
                missing.append('preview identity mismatch:' + str(p.relative_to(root)))
        for key, value in spec.items():
            if key.endswith('path') and isinstance(value, str):
                target = mapped(value)
                if target and not target.is_file():
                    missing.append(value)
    return {'databases': databases, 'missing_references': missing,
            'incompatible_paths_or_schema': incompatible}


def restore(args):
    dest = args.destination.resolve()
    dest.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(args.archive, 'r:*') as archive:
        names = set()
        for member in archive:
            name = PurePosixPath(member.name)
            if (name.is_absolute() or '..' in name.parts or str(name) in names
                    or not member.isfile()
                    or (name.parts[0] != 'data' and str(name) != 'manifest.json')):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(str(name))
            target = dest / str(name)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open('xb') as f, archive.extractfile(member) as source:
                while block := source.read(1024 * 1024):
                    f.write(block)
            target.chmod(0o600)
    manifest = json.loads((dest / 'manifest.json').read_text())
    root = dest / 'data'
    if set(manifest['files']) != {str(p.relative_to(root)) for p in files(root, set())}:
        raise ValueError('Archive file inventory mismatch')
    for name, expected in manifest['files'].items():
        target = root / name
        if target.stat().st_size != expected['size'] or digest(target) != expected['sha256']:
            raise ValueError('Archive hash mismatch: ' + name)
    report = inventory(root, manifest['source_root'])
    report.update(archive_sha256=digest(args.archive), files_verified=len(manifest['files']),
                  consistency=manifest['consistency'])
    (dest / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if any(d['integrity'] != 'ok' or d['foreign_keys'] for d in report['databases'].values()):
        raise ValueError('Restored database validation failed')
    if report['missing_references'] or report['incompatible_paths_or_schema']:
        raise ValueError('Restored files need migration review; do not start the app')


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('snapshot', help='writes sensitive TAR to stdout')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--exclude', action='append', default=[], help='exact top-level directory name')
    p.add_argument('--consistency', choices=['online-rehearsal', 'stopped-writers'],
                   default='online-rehearsal')
    p.set_defaults(func=snapshot)
    p = commands.add_parser('restore', help='hash + SQLite + file relationship verification')
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.set_defaults(func=restore)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
