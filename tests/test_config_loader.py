"""Unit tests for kinetix.configs.config_loader (path + API-key resolution).

Self-contained: every test resolves paths through the ``KINETIX_DATA_DIR``
override into ``tmp_path`` fixtures; no repository data files are touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kinetix.configs import config_loader as cl


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  """Isolate every test from ambient KINETIX_DATA_DIR / MP_API_KEY."""
  monkeypatch.delenv(cl.ENV_DATA_DIR, raising=False)
  monkeypatch.delenv('MP_API_KEY', raising=False)


def expected_repo_root() -> Path:
  return Path(cl.__file__).resolve().parents[2]


class TestProjectRoot:
  def test_default_is_repo_root(self):
    assert cl.get_project_root() == expected_repo_root()

  def test_env_override(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    assert cl.get_project_root() == tmp_path

  def test_data_subdirs_follow_root(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    assert cl.get_data_root() == tmp_path / 'data'
    assert cl.get_parameters_root() == tmp_path / 'data' / 'parameters'
    assert cl.get_grids_root() == tmp_path / 'data' / 'grids'
    assert cl.get_mesh_root() == tmp_path / 'data' / 'mesh'

  def test_config_path_follows_root(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    assert cl.get_config_path() == tmp_path / 'config.json'


class TestLoadConfig:
  def test_valid_json(self, tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'api_key': 'secret', 'extra': 1}))
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    assert cl.load_config() == {'api_key': 'secret', 'extra': 1}

  def test_missing_file_mentions_env_var(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    with pytest.raises(FileNotFoundError, match='MP_API_KEY'):
      cl.load_config()

  def test_missing_api_key_raises_key_error(self, tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({}))
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    with pytest.raises(KeyError, match='api_key'):
      cl.load_config()


class TestApiKey:
  def test_env_takes_priority_over_file(self, tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'api_key': 'file-key'}))
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    monkeypatch.setenv('MP_API_KEY', '  env-key  ')
    assert cl.get_api_key() == 'env-key'  # env wins, and is stripped

  def test_env_works_without_config_file(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))  # no config.json here
    monkeypatch.setenv('MP_API_KEY', 'env-key')
    assert cl.get_api_key() == 'env-key'

  def test_falls_back_to_config_file(self, tmp_path, monkeypatch):
    (tmp_path / 'config.json').write_text(json.dumps({'api_key': 'file-key'}))
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    assert cl.get_api_key() == 'file-key'


class TestActivationEnergies:
  def test_loads_json_relative_to_presets_parent(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    presets_dir = tmp_path / 'parameters' / 'presets'
    presets_dir.mkdir(parents=True)
    ae_path = tmp_path / 'parameters' / 'ae_test.json'
    ae_path.write_text(json.dumps({'PZT': [{'specie': 'H', 'E_gen': 1.0}]}))
    settings = SimpleNamespace(activation_energies='ae_test.json')
    result = cl.load_activation_energies(presets_dir / 'preset.yaml', settings)
    assert result == {'PZT': [{'specie': 'H', 'E_gen': 1.0}]}

  def test_missing_file_raises(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    presets_dir = tmp_path / 'parameters' / 'presets'
    presets_dir.mkdir(parents=True)
    settings = SimpleNamespace(activation_energies='nope.json')
    with pytest.raises(FileNotFoundError):
      cl.load_activation_energies(presets_dir / 'preset.yaml', settings)

  def test_unset_setting_raises_value_error(self, tmp_path, monkeypatch):
    monkeypatch.setenv(cl.ENV_DATA_DIR, str(tmp_path))
    settings = SimpleNamespace(activation_energies=None)
    with pytest.raises(ValueError, match='activation energies'):
      cl.load_activation_energies(tmp_path / 'preset.yaml', settings)
