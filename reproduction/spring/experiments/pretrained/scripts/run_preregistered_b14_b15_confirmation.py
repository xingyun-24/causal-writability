#!/usr/bin/env python3
"""Confirm B14 writeability and B15 closure on a frozen curriculum cohort."""
from __future__ import annotations
import argparse,gc,json,math,sys
from collections import Counter
from pathlib import Path
from statistics import median
import yaml,torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pretrained_wan_pca_causal_controller import load_model,metric,safe_path,sample
from sshv2.interpretability.condition_residual_patching import ResidualPatchController
from sshv2.interpretability.qualified_pair_bank_128_runtime import encode_long_condition,load_frames_npz
from sshv2.simulation.spring_shortcuts_v1 import apply_short_history_mask_numpy,dataclass_config_from_dict
STEPS=20
def args():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-config',type=Path,required=True);p.add_argument('--fast-bank',type=Path,required=True);p.add_argument('--slow-bank',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--layers',type=int,nargs='+',default=[14,15]);p.add_argument('--direction-filter',choices=('fast','slow'));p.add_argument('--first-observed-filter',type=int);return p.parse_args()
def atomic(path,payload):path.parent.mkdir(parents=True,exist_ok=True);t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(payload,indent=2)+'\n');t.replace(path)
def main():
 a=args();a.out.mkdir(parents=True,exist_ok=True);rd=a.out/'records';rd.mkdir(exist_ok=True);cfg=dataclass_config_from_dict(yaml.safe_load(a.data_config.read_text()));pipe,cfg2,n=load_model(a);assert cfg==cfg2
 indices={}
 for direction,root in [('fast',a.fast_bank),('slow',a.slow_bank)]:indices[direction]={json.loads(x)['pair_id']:json.loads(x) for x in (root/'qualified_pairs_128.jsonl').read_text().splitlines() if x}
 selected=[json.loads(x) for x in a.selection.read_text().splitlines() if x];selected=[x for x in selected if (a.direction_filter is None or x['direction']==a.direction_filter) and (a.first_observed_filter is None or x.get('first_observed_shortcut')==a.first_observed_filter)];layers=tuple(a.layers);all=[]
 for i,item in enumerate(selected,1):
  direction=item['direction'];root=a.fast_bank if direction=='fast' else a.slow_bank;row=indices[direction][item['pair_id']];dst=rd/f'{direction}_{row["pair_id"]}.json'
  if a.resume and dst.is_file():all.append(json.loads(dst.read_text()));continue
  seed=int(row['generation_seed']);color='red' if direction=='fast' else 'blue';conds={};metas={}
  for role in ('aligned','conflict'):
   frames=apply_short_history_mask_numpy(load_frames_npz(safe_path(root,row[f'{role}_input_frames_npz'])),cfg);conds[role]=encode_long_condition(pipe,frames,prediction_start=cfg.prediction_start,num_condition_frames=n);metas[role]=json.loads(safe_path(root,row[f'{role}_metadata']).read_text())
  donor=ResidualPatchController(STEPS,n,record_condition_layers=layers);lat=sample(pipe,conds['aligned'],cfg,n,seed,donor);aligned=metric(pipe,lat,cfg,metas['aligned'],'blue' if direction=='fast' else 'red');del lat
  natural_ctl=ResidualPatchController(STEPS,n,record_condition_layers=layers);lat=sample(pipe,conds['conflict'],cfg,n,seed,natural_ctl);conflict=metric(pipe,lat,cfg,metas['conflict'],color);del lat
  oa=aligned.get('omega_pixel');oc=conflict.get('omega_pixel');den=(float(oa)-float(oc)) if oa is not None and oc is not None else None;runs=[]
  for layer in layers:
   ctl=ResidualPatchController(STEPS,n,inject_bank=donor.bank,inject_layer=layer,inject_kind='condition');lat=sample(pipe,conds['conflict'],cfg,n,seed,ctl);res=metric(pipe,lat,cfg,metas['conflict'],color);op=res.get('omega_pixel');R=((float(op)-float(oc))/den if op is not None and den is not None and abs(den)>1e-8 else None);runs.append({'layer':layer,'R':R,**res});del ctl,lat
  edit_audit={}
  for layer in layers:
   donor_steps=donor.bank.condition[layer];conflict_steps=natural_ctl.bank.condition[layer];sq=base=0.0;count=0
   for left,right in zip(donor_steps,conflict_steps,strict=True):
    delta=left.float()-right.float();sq+=float(torch.sum(delta*delta,dtype=torch.float64));base+=float(torch.sum(right.float()*right.float(),dtype=torch.float64));count+=delta.numel()
   edit_audit[str(layer)]={'rms':math.sqrt(sq/count),'conflict_rms':math.sqrt(base/count),'relative_rms':math.sqrt(sq/base) if base>0 else None}
  record={'trajectory_id':row['trajectory_id'],'pair_id':row['pair_id'],'direction':direction,'selection_reason':item.get('selection_reason'),'first_observed_shortcut':item.get('first_observed_shortcut'),'aligned':aligned,'conflict':conflict,'matched_edit_audit':edit_audit,'runs':runs};atomic(dst,record);all.append(record);print(f'[{i}/{len(selected)}] {direction}',flush=True);gc.collect()
 groups={}
 for layer in layers:
  for direction in ('pooled','fast','slow'):
   z=[run for rec in all if direction=='pooled' or rec['direction']==direction for run in rec['runs'] if run['layer']==layer];vals=[float(x['R']) for x in z if x.get('R') is not None and math.isfinite(float(x['R']))];groups[f'{direction}/B{layer}']={'n':len(z),'finite_R_n':len(vals),'median_R':median(vals) if vals else None,'strong_R_rate':sum(.75<x<1.25 for x in vals)/len(z) if z else None,'valid_rate':sum(bool(x.get('valid')) for x in z)/len(z) if z else None,'route_counts_valid':dict(Counter(str(x.get('route_label')) for x in z if x.get('valid')))}
 atomic(a.out/'summary.json',{'status':'PASS','checkpoint':str(a.checkpoint),'selection':str(a.selection),'n':len(all),'layers':list(layers),'direction_filter':a.direction_filter,'first_observed_filter':a.first_observed_filter,'groups':groups,'interpretation':('confirmatory matched condition-prefix replacement; no layer selection on this cohort' if layers==(14,15) else 'pre-registered local fallback after B14/B15 heterogeneity; no full-network layer search')});print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
