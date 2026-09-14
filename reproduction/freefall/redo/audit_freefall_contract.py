#!/usr/bin/env python3
"""Validate the Free-fall rank-2 release package and its manifest."""
import argparse,csv,hashlib,json
from pathlib import Path

REQUIRED=['manifest.json','resolved_config.yaml','checkpoint_sha256.txt','code_commit.txt','boundary_match_audit.csv','evaluator_contract.json','evaluator_audit.csv','behavior_rollouts.csv','behavior_summary.json','strict_bank.csv','split_manifest.json','matched_directions.npy','pca_components.npy','pca_summary.json','coordinate_predictions.csv','coordinate_summary.json','decoded_recovery.csv','decoded_recovery_summary.json','layer_scan.csv','layer_summary.json','video_registry.jsonl','QA.md']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rows(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser();p.add_argument('kit',type=Path);a=p.parse_args();k=a.kit
    missing=[x for x in REQUIRED if not (k/x).is_file()]
    for stem in ('freefall_behavior','freefall_state_geometry','freefall_decoded_recovery','freefall_writeability'):
        missing += [f'figures/{stem}.{ext}' for ext in ('svg','pdf','png') if not (k/f'figures/{stem}.{ext}').is_file()]
    assert not missing,missing
    manifest=json.loads((k/'manifest.json').read_text());listed={r['path']:r for r in manifest['files']}
    actual={str(x.relative_to(k)).replace('\\','/') for x in k.rglob('*') if x.is_file() and x.name!='manifest.json'}
    assert actual==set(listed),(actual-set(listed),set(listed)-actual)
    for rel,item in listed.items():
        path=k/rel;assert path.stat().st_size==item['bytes'];assert sha(path)==item['sha256']
    behavior=rows(k/'behavior_rollouts.csv');strict=rows(k/'strict_bank.csv');coord=rows(k/'coordinate_predictions.csv');recovery=rows(k/'decoded_recovery.csv');layer=rows(k/'layer_scan.csv');boundary=rows(k/'boundary_match_audit.csv')
    assert len(behavior)==1408 and len(strict)==128 and len(coord)==256 and len(recovery)==256 and len(layer)==1920 and len(boundary)==128
    assert all(r['pass']=='True' for r in boundary)
    assert {r['condition'] for r in recovery}=={'natural_conflict','full_matched_edit','top2_oracle_projection','fit_only_controller'}
    assert all(sum(r['condition']==condition for r in recovery)==64 for condition in {r['condition'] for r in recovery})
    # F1 retains invalid futures in the denominator by design; F3 must be valid.
    assert all(r['valid']=='True' for r in recovery)
    print(json.dumps({'status':'PASS','manifest_files':len(actual),'behavior_rows':len(behavior),'strict_pairs':len(strict),'coordinate_rows':len(coord),'recovery_rows':len(recovery),'layer_rows':len(layer),'boundary_passes':len(boundary)},indent=2))

if __name__=='__main__':main()
