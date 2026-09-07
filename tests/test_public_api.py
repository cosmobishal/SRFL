from srfl import __version__


def test_version_is_semverish():
    parts = __version__.split('.')
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
