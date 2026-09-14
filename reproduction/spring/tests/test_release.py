import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from causal_writability.attention import ConditionQKVController
from causal_writability.edit import choose_rows, phase_features
from causal_writability.physics import candidate_definition
from causal_writability.weights import digest, install
from sshv2.interpretability.condition_residual_patching import ResidualPatchController, ActivationBank
from sshv2.simulation.spring_shortcuts_v1 import dataclass_config_from_dict

ROOT = Path(__file__).resolve().parents[1]


def test_model_ids_are_unique():
    rows = json.loads((ROOT / 'models/registry.json').read_text())
    assert len({r['id'] for r in rows}) == len(rows)
    assert len(rows) == 7
    assert all('sha256' not in row for row in rows)


def test_private_hf_download_uses_logged_in_client(tmp_path, monkeypatch):
    from causal_writability import weights
    import huggingface_hub
    cached = tmp_path / 'cached-model'
    cached.write_bytes(b'test model')
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps([{'id': 'test-model', 'hf_repo': 'owner/models', 'hf_filename': 'spring/model.safetensors'}]))
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    target = tmp_path / 'installed'
    monkeypatch.setattr('sys.argv', ['weights', '--registry', str(registry), '--model', 'test-model', '--out', str(target)])
    weights.main()
    assert calls == [{'repo_id': 'owner/models', 'filename': 'spring/model.safetensors'}]
    assert target.read_bytes() == cached.read_bytes()


def test_weight_install_is_atomic(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'test-only model bytes')
    target = tmp_path / 'target'
    with pytest.raises(ValueError):
        install(str(source), target, 'bad-digest')
    assert not target.exists()
    assert install(str(source), target, digest(source)) == digest(source)
    with pytest.raises(FileExistsError):
        install(str(source), target)


def test_frozen_receivers_and_split():
    bank = json.loads((ROOT / 'data/strict_bank.json').read_text())
    split = json.loads((ROOT / 'data/strict_split.json').read_text())
    assert len(bank) == len(split) == 256
    fit = {r['trajectory_id'] for r in split if r['split'] == 'fit'}
    held = {r['trajectory_id'] for r in split if r['split'] == 'held_out'}
    assert len(fit) == len(held) == 128 and not fit & held
    cfg = dataclass_config_from_dict(yaml.safe_load((ROOT / 'configs/data.yaml').read_text()))
    for direction in ['true_fast_red_conflict', 'true_slow_blue_conflict']:
        rows = choose_rows(bank, split, direction, 0)
        assert len(rows) == 64
        row = rows[0]
        generated = candidate_definition(direction=direction, candidate_index=row['candidate_index'], cfg=cfg)
        assert generated['trajectory_id'] == row['trajectory_id']
        for key in ['omega_true', 'amplitude', 'phase', 'x_star', 'v_star']:
            assert generated[key] == pytest.approx(row[key], abs=1e-12)
        assert phase_features(row).shape == (6,)


def test_residual_patch_leaves_future_untouched():
    source = torch.ones(1, 4, 8)
    bank = ActivationBank()
    bank.record('condition', 0, source)
    ctl = ResidualPatchController(1, 1, inject_bank=bank, inject_layer=0)
    x = torch.zeros(1, 8, 8)
    result = ctl.apply(x, layer=0, f=2, h=2, w=2)
    assert torch.equal(result[:, :4], source)
    assert torch.equal(result[:, 4:], x[:, 4:])


def test_attention_head_and_future_scope():
    ctl = ConditionQKVController(1, 1, 2)
    raw = torch.ones(1, 8, 12)
    for name in ['q', 'k', 'v']:
        ctl._hook(name)(None, None, raw)
    ctl.finish()
    inject = ConditionQKVController(1, 1, 2, mode='inject', components=('v',),
                                    source=ctl.bank, head_indices=(1,), num_heads=3, blend_alpha=2)
    zero = torch.zeros_like(raw)
    for name in ['q', 'k', 'v']:
        result = inject._hook(name)(None, None, zero)
        assert torch.equal(result[:, 4:], zero[:, 4:])
        if name == 'v':
            assert torch.all(result[:, :4, 4:8] == 2)
            assert torch.all(result[:, :4, :4] == 0)
            assert torch.all(result[:, :4, 8:] == 0)
        else:
            assert torch.equal(result, zero)
    inject.finish()


def test_no_private_paths_in_source():
    for path in (ROOT / 'src').rglob('*.py'):
        text = path.read_text()
        assert '/data1/' not in text and '/data2/' not in text and '/data3/' not in text and '/data4/' not in text, path
    for path in (ROOT / 'configs').glob('*.yaml'):
        assert '/data' not in path.read_text(), path
