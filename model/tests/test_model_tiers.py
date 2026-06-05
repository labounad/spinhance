"""
Enforces the production model-size *tier* convention (the single source of truth
for the data-scaling fleet):

    light -> 64k data   (~10M params)
    med   -> 500k data  (~57M params)
    xl    -> 3M data     (~137M params)

`TIER_PRESETS` (model/architectures/resnet1d.py) defines each tier fully (conv stem
+ transformer width/depth); fleet configs select a tier via `model.size` and set
nothing else size-related. These tests guard against drift: the tier param counts,
tier-takes-precedence + raw-stem back-compat, and that every fleet config under
model/configs/ (train_64k_* / train_500k_* / train_3M_*) actually uses the right
tier name and stays size-only.
"""
import glob
import os

import pytest
import yaml

from model.architectures import build_architecture
from model.architectures.resnet1d import SIZE_PRESETS, TIER_PRESETS

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

# data-tier filename prefix -> (tier name, expected ~params in millions)
FLEET = {
    "train_64k_0":  ("light", 10.0),
    "train_500k_0": ("med",   56.6),
    "train_3M_0":   ("xl",    137.4),
}
# size-related keys that must NOT appear in a fleet config (the tier owns them)
SIZE_KEYS = {"dim", "enc_layers", "dec_layers", "n_heads"}


def test_tier_presets_exist():
    assert set(TIER_PRESETS) == {"light", "med", "xl"}
    for spec in TIER_PRESETS.values():
        assert spec["stem"] in SIZE_PRESETS          # resolves to a real conv stem
        assert {"stem", "dim", "enc_layers", "dec_layers", "n_heads"} <= set(spec)


def test_light_tier_param_count():
    m = build_architecture("spingraph_decoder", size="light")
    assert 9.0e6 < m.n_params < 11.0e6


def test_tier_takes_precedence_and_back_compat():
    # tier name resolves to the tier's transformer dim (768 for xl)...
    xl = build_architecture("spingraph_decoder", size="xl")
    assert xl.enc.layers[0].linear1.in_features == 768
    # ...while a raw conv-stem preset still works (back-compat) at the legacy default dim
    raw = build_architecture("spingraph_decoder", size="medium")
    assert raw.enc.layers[0].linear1.in_features == 256
    # and an explicit dim overrides the tier
    ov = build_architecture("spingraph_decoder", size="light", dim=128)
    assert ov.enc.layers[0].linear1.in_features == 128


@pytest.mark.slow
@pytest.mark.parametrize("tier,exp_m", [("med", 56.6), ("xl", 137.4)])
def test_large_tier_param_counts(tier, exp_m):
    m = build_architecture("spingraph_decoder", size=tier)
    assert abs(m.n_params / 1e6 - exp_m) < 1.0, f"{tier}: {m.n_params/1e6:.2f}M != ~{exp_m}M"


def _fleet_configs():
    out = []
    for prefix, (tier, _) in FLEET.items():
        for path in sorted(glob.glob(os.path.join(CONFIG_DIR, prefix + "*.yaml"))):
            out.append((os.path.basename(path), path, tier))
    return out


def test_fleet_configs_found():
    # sanity: the rename landed — we have the three data tiers present
    names = [n for n, _, _ in _fleet_configs()]
    assert any(n.startswith("train_64k_0") for n in names)
    assert any(n.startswith("train_500k_0") for n in names)
    assert any(n.startswith("train_3M_0") for n in names)
    # the mis-named / stale configs are gone
    assert not glob.glob(os.path.join(CONFIG_DIR, "train_3M_spingraph_*.yaml"))
    assert not glob.glob(os.path.join(CONFIG_DIR, "train_500k_light_*.yaml"))


@pytest.mark.parametrize("name,path,tier", _fleet_configs())
def test_fleet_config_uses_correct_tier_and_is_size_only(name, path, tier):
    model = yaml.safe_load(open(path))["model"]
    assert model.get("size") == tier, f"{name}: size={model.get('size')!r}, expected {tier!r}"
    leaked = SIZE_KEYS & set(model)
    assert not leaked, f"{name}: sets {leaked} — fleet configs must be size-only (tier owns size)"
