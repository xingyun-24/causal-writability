"""Check the collected corrected tables without loading a model or changing them."""
from pathlib import Path
import contextlib
import csv
import importlib.util
import io
import json
import math
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'paper/pendulum'


def rows(name):
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    elif isinstance(expected, float):
        # Python versions differ in floating-point summation by a few ULPs.
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    else:
        assert actual == expected


def main():
    behavior = rows('behavior_rollouts.csv')
    assert len(behavior) == 2816
    coordinates = rows('coordinate_predictions.csv')
    assert len(coordinates) == 128
    for direction in ['low', 'high']:
        group = [r for r in coordinates if r['target_label'] == direction]
        assert sum(r['split'] == 'fit' for r in group) == 32
        assert sum(r['split'] == 'heldout' for r in group) == 32
    recovery = rows('decoded_recovery.csv')
    summary = json.loads((DATA / 'decoded_recovery_summary.json').read_text())
    report = {}
    for condition in ['full_matched', 'top4_oracle', 'fit_only_predicted']:
        group = [r for r in recovery if r['condition'] == condition]
        assert len(group) == 64 and all(r['valid'].lower() == 'true' for r in group)
        values = [float(r['normalized_recovery']) for r in group]
        record = summary['conditions'][condition]
        assert np.isclose(np.median(values), record['conditional_median_recovery'], atol=1e-12)
        assert sum(.75 < value < 1.25 for value in values) == record['near_full_success_n'] == 63
        if condition == 'top4_oracle':
            assert all('/edit_rank_4.mp4' in r['output_video_path'] for r in group)
        report[condition] = {'n': len(values), 'median_R': float(np.median(values)), 'near_full': 63}
    spec = importlib.util.spec_from_file_location('probability_check', DATA / 'prepare_pendulum_probability_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='pendulum-tables-') as temp:
        module.HERE = Path(temp)
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
        assert_same(json.loads((module.HERE / 'layer_summary.json').read_text()), json.loads((DATA / 'layer_summary.json').read_text()))
        with (module.HERE / 'layer_scan.csv').open() as f:
            assert list(csv.DictReader(f)) == rows('layer_scan.csv')
    print(json.dumps({'behavior_rows': len(behavior), 'coordinate_rows': len(coordinates),
                      'recovery': report, 'layer_tables_match_recomputed_evaluator_v2': True}, indent=2))


if __name__ == '__main__':
    main()
