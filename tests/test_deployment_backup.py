import io
import json
import sqlite3
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/deployment-data.py'


@pytest.mark.parametrize("consistency", ["online-rehearsal", "stopped-writers"])
def test_wal_backup_and_restore_preserve_media_secrets(tmp_path, consistency):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.maintenance-token').write_text('fixture-secret-only')
    (source / 'uploads').mkdir()
    media = source / 'uploads' / 'fixture.mp4'
    media.write_bytes(b'fixture-media')
    c = sqlite3.connect(source / 'state.sqlite3')
    c.executescript('''pragma journal_mode=wal;
        create table uploads(path text,size int,offset int,status text,id text);
        create table jobs(id text,upload_id text,status text,result_json text);
    ''')
    c.execute('insert into uploads values(?,?,?,?,?)',
              (str(media), media.stat().st_size, media.stat().st_size, 'completed', 'test'))
    c.execute("insert into jobs values('test','test','completed',null)")
    c.commit()
    archive = tmp_path / 'backup.tar'
    with archive.open('wb') as out:
        subprocess.run([sys.executable, SCRIPT, 'snapshot', '--source', source,
                        '--consistency', consistency], stdout=out,
                       check=True)
    c.close()
    dest = tmp_path / 'restored'
    result = subprocess.run([sys.executable, SCRIPT, 'restore', '--archive', archive,
                             '--destination', dest], capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report['files_verified'] == 3
    assert report['databases']['state.sqlite3']['integrity'] == 'ok'
    assert (dest / 'data/.maintenance-token').read_text() == 'fixture-secret-only'
    assert (dest / 'data/uploads/fixture.mp4').read_bytes() == b'fixture-media'
    assert (dest / 'data/.maintenance-token').stat().st_mode & 0o777 == 0o600
    repeated = subprocess.run([sys.executable, SCRIPT, 'restore', '--archive', archive,
                              '--destination', dest], capture_output=True)
    assert repeated.returncode != 0


def test_restore_refuses_path_traversal(tmp_path):
    archive = tmp_path / 'unsafe.tar'
    with tarfile.open(archive, 'w') as t:
        member = tarfile.TarInfo('data/../../escaped')
        member.size = 1
        t.addfile(member, io.BytesIO(b'x'))
    result = subprocess.run([sys.executable, SCRIPT, 'restore', '--archive', archive,
                             '--destination', tmp_path / 'restored'], capture_output=True)
    assert result.returncode != 0
    assert not (tmp_path / 'escaped').exists()
