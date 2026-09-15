from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from pingpong_highlight.config import Settings
from pingpong_highlight.web import create_app


@pytest.mark.parametrize('prefix', ['', '/pingpong-highlight'])
def test_external_urls_cookie_scope_and_private_cache(tmp_path, prefix):
    settings = Settings(
        data_dir=tmp_path, upload_token='unused-test-token', root_path=prefix,
        public_url='https://clips.example.test' + prefix,
        session_cookie_secure=True, bootstrap_admin_password='test-password-123',
        allowed_hosts=('clips.example.test',),
    )
    app = create_app(settings)
    with TestClient(app, base_url='https://clips.example.test') as client:
        html = client.get(prefix + '/')
        assert html.status_code == 200
        assert f'content="{prefix}"' in html.text
        assert f'src="{prefix}/static/paths.js"' in html.text
        for path in ['/static/index.html', '/static/review/index.html']:
            response = client.get(prefix + path)
            assert '__HC_ROOT_PATH__' not in response.text
            assert f'content="{prefix}"' in response.text
        assert client.get(prefix + '/api/jobs').headers['cache-control'] == 'private, no-store'
        login = client.post(prefix + '/api/auth/login', json={
            'username': 'admin', 'password': 'test-password-123',
        })
        assert login.status_code == 200
        cookie = login.headers['set-cookie']
        assert f'{settings.session_cookie_name}=' in cookie
        assert f'Path={prefix}/;' in cookie
        assert 'Secure' in cookie and 'HttpOnly' in cookie and 'SameSite=strict' in cookie
        assert client.get(prefix + '/api/auth/me').status_code == 200
        upload = client.post(prefix + '/api/uploads', headers={
            'Tus-Resumable': '1.0.0', 'Upload-Length': '3',
            'Upload-Metadata': 'filename dGVzdC5tcDQ=',
        })
        assert upload.status_code == 201
        assert upload.headers['location'].startswith(prefix + '/api/uploads/')
        head = client.head(upload.headers['location'], headers={'Tus-Resumable': '1.0.0'})
        assert head.status_code == 200 and head.headers['upload-offset'] == '0'
        assert client.get(prefix + '/static/app.js').headers['cache-control'] == 'no-cache'
        assert client.get(prefix + '/api/health', headers={'Host': 'evil.test'}).status_code == 400
        logout = client.post(prefix + '/api/auth/logout')
        assert 'Max-Age=0' in logout.headers['set-cookie']
        assert f'Path={prefix}/;' in logout.headers['set-cookie']
        assert client.get(prefix + '/api/auth/me').status_code == 401
        if prefix:
            assert 'clear-site-data' not in logout.headers
            assert settings.session_cookie_name != 'pingpong_session'


@pytest.mark.parametrize('prefix', ['/trailing/', '//double', '/bad%20path', '/a/../b', 'relative'])
def test_reject_ambiguous_prefix(tmp_path, prefix):
    with pytest.raises(ValueError, match='root_path'):
        Settings(data_dir=tmp_path, upload_token='test', root_path=prefix)


def test_reject_public_url_path_mismatch_and_cookie_collision(tmp_path):
    settings = Settings(data_dir=tmp_path, upload_token='test')
    with pytest.raises(ValueError, match='public_url'):
        replace(settings, public_url='https://example.test/pingpong-highlight')
    with pytest.raises(ValueError, match='__Host-'):
        replace(settings, root_path='/pingpong-highlight', session_cookie_secure=True,
                session_cookie_name='__Host-highlightcraft')
