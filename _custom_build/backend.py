from setuptools import build_meta as _orig

prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist


def get_requires_for_build_wheel(config_settings=None):
    return _orig.get_requires_for_build_wheel(config_settings) + [...]


def get_requires_for_build_sdist(config_settings=None):
    return _orig.get_requires_for_build_sdist(config_settings) + [...]