#!/usr/bin/env python3
"""Build the audited Free-fall paper/data package for train-only PCA rank 2."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,shutil
from collections import defaultdict
from pathlib import Path

import cv2,imageio.v2 as imageio,matplotlib as mpl,matplotlib.pyplot as plt,numpy as np

LOW='#D66555'; HIGH='#4F75B3'; PURPLE='#9575B5'; TEAL='#3A9688'; AMBER='#D19A45'; INK='#30343B'; SECONDARY='#899097'; GRID='#DDE0E2'; INVALID='#C8CBCB'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))
def write_json(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def configure():
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Helvetica','Arial','Liberation Sans','DejaVu Sans'],'font.size':7,
        'axes.titlesize':8.5,'axes.titleweight':'semibold','axes.labelsize':7.5,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.5,
        'axes.edgecolor':INK,'axes.labelcolor':INK,'xtick.color':'#626970','ytick.color':'#626970','text.color':INK,'axes.facecolor':'white','figure.facecolor':'white',
        'axes.grid':True,'grid.color':GRID,'grid.linewidth':.5,'grid.alpha':.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','savefig.bbox':None,'savefig.pad_inches':.04})


def save_figure(fig,out,stem):
    fig.savefig(out/f'{stem}.svg');fig.savefig(out/f'{stem}.pdf');fig.savefig(out/f'{stem}.png',dpi=600);plt.close(fig)


def track(frames,radius=6):
    ys=np.full(len(frames),np.nan); rgbs=np.full((len(frames),3),np.nan); prior=None
    for i,frame in enumerate(frames.astype(np.uint8)):
        hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV);mask=((hsv[...,1]>=45)&(hsv[...,2]>=55)).astype(np.uint8)
        count,labels,stats,centres=cv2.connectedComponentsWithStats(mask,8);cand=[]
        for label in range(1,count):
            area=int(stats[label,cv2.CC_STAT_AREA])
            if not 28<=area<=420:continue
            x,y=centres[label];temporal=0 if prior is None else .2*math.hypot(x-prior[0],y-prior[1]);cand.append((abs(area-math.pi*radius**2)*.02+temporal,label))
        if not cand:continue
        _,label=min(cand);x,y=centres[label];component=labels==label;ys[i]=1-y/(frame.shape[0]-1);rgbs[i]=frame[component].mean(0);prior=(x,y)
    return ys,rgbs,np.isfinite(ys)


def audit_frames(frames,row):
    y,rgb,valid=track(frames);t=(np.arange(len(y))+1)/20.;yb=json.loads(row['position_at_boundary'])[1];vy=json.loads(row['velocity_at_boundary'])[1]
    target=y-yb-vy*t;design=np.column_stack((np.ones_like(t),t,-.5*t*t));beta=np.linalg.lstsq(design[valid],target[valid],rcond=None)[0]
    g=float(beta[-1]);fit=yb+vy*t+design@beta;g0=float(np.linalg.lstsq((-.5*t[valid]**2)[:,None],target[valid],rcond=None)[0][0]);med=np.nanmedian(rgb,axis=0)
    miss=best=0
    for v in valid:miss=0 if v else miss+1;best=max(best,miss)
    return {'gravity_hat_e3':g,'gravity_hat_e0':g0,'fit_rmse':float(np.sqrt(np.mean((y[valid]-fit[valid])**2))),
            'valid':bool(valid.sum()>=58),'invalid_reasons':'' if valid.sum()>=58 else 'detected_frames_below_58','detected_frames':int(valid.sum()),
            'detector_coverage':float(valid.mean()),'max_missing_run':int(best),'max_adjacent_y_jump':float(np.nanmax(np.abs(np.diff(y)))) if len(y)>1 else None,
            'future_r':float(med[0]),'future_g':float(med[1]),'future_b':float(med[2]),'future_color_coordinate':float((med[2]-med[0])/(med[2]+med[0]+1e-12)),
            'detected_colour':'red' if med[0]>med[2] else 'blue'}


def to_mp4(npz_path,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);frames=np.load(npz_path)['frames'];imageio.mimsave(dest,frames,fps=20,codec='libx264',ffmpeg_params=['-preset','ultrafast','-crf','0','-pix_fmt','yuv444p']);return frames


def find_frame(root,rank,pid):
    paths=list(root.glob(f'shard*/frames/{rank}/{pid}.npz'));assert len(paths)==1,(rank,pid,paths);return paths[0]


def find_controller_frame(root,pid,condition):
    paths=list(root.glob(f'shard*/frames/{pid}__{condition}.npz'));assert len(paths)==1,(pid,condition);return paths[0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--old-kit',type=Path,required=True);p.add_argument('--f1-root',type=Path,required=True);p.add_argument('--basis',type=Path,required=True);p.add_argument('--aligned-reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    work=a.work;out=a.out
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    for d in ('figures','scripts','videos/behavior','videos/recovery','videos/posters','data'): (out/d).mkdir(parents=True,exist_ok=True)
    configure()
    metadata_rows=list(csv.DictReader((work/'delivery_inputs/metadata.csv').open(newline='',encoding='utf-8')));meta={(r['pair_id'],r['variant']):r for r in metadata_rows}
    pairs=read_json(work/'delivery_inputs/recovery_pairs_strict128.json')['pairs'];pair_map={r['pair_id']:r for r in pairs}
    evals={(r['pair_id'],r['condition']):r for r in read_json(work/'delivery_inputs/eval_rows_merged.json')}
    split=read_json(work/'results-grouped/split_manifest.json');train=set(split['train_pair_ids']);heldout=set(split['heldout_pair_ids'])
    assert len(train)==len(heldout)==64 and train.isdisjoint(heldout)
    boundary=list(csv.DictReader((work/'delivery_inputs/boundary_match_audit.csv').open(newline='',encoding='utf-8')));assert len(boundary)==128 and all(r['pass']=='True' for r in boundary);boundary_by={r['pair_id']:r for r in boundary}
    shutil.copy2(work/'delivery_inputs/boundary_match_audit.csv',out/'boundary_match_audit.csv')
    shutil.copy2(a.old_kit/'resolved_config.yaml',out/'resolved_config.yaml');shutil.copy2(a.old_kit/'data_v3_fixedpos_freefall.yaml',out/'data_v3_fixedpos_freefall.yaml')
    (out/'checkpoint_sha256.txt').write_text('f2fb7275c3385a910cb498c620ce2f72842d46773c3f25a76634109937bec11d  step-100000.safetensors\n')
    (out/'code_commit.txt').write_text('physics-shortcuts-benchmarks HEAD 8386c6f0f7d23f38d1e74157da61c0ba47c246db; working tree dirty; delivery scripts identified by manifest SHA256\n')
    controller=read_json(work/'delivery_inputs/directional_controller_fit.json');write_json(out/'directional_controller_fit.json',controller)
    packed=np.load(work/'results-grouped/train_only_pca.npz');basis=np.load(a.basis,mmap_mode='r');assert basis.shape==(2,20,1088,1536)
    flat=basis.reshape(2,-1);component_gram=np.zeros((2,2))
    for start in range(0,flat.shape[1],1_000_000):
        block=flat[:,start:start+1_000_000].astype(np.float64);component_gram+=block@block.T
    np.testing.assert_allclose(component_gram,np.eye(2),rtol=1e-7,atol=1e-7);np.save(out/'pca_components.npy',basis)
    np.save(out/'matched_directions.npy',np.asarray(['target_low','target_high']))
    np.savez_compressed(out/'pca_basis_metadata.npz',pair_ids=packed['pair_ids'],train_mask=packed['train_mask'],heldout_mask=packed['heldout_mask'],eigenvalues=packed['eigenvalues'],train_vectors=packed['train_vectors'],scores=packed['scores'],delta_shape=packed['delta_shape'])
    pca_source=read_json(work/'results-grouped/summary.json');pca_summary={'method':'uncentered SVD/PCA of aligned-minus-conflict Block 1 condition-prefix residuals; fit split only','rank':2,'fit_pairs':64,'heldout_pairs':64,
        'split_counts':pca_source['split_counts'],'rank_selection':'smallest fit-only rank retaining at least 99% of training residual energy','train_retained_energy_fraction':pca_source['rank_metrics']['2']['train']['retained_energy_fraction'],
        'heldout_retained_energy_fraction':pca_source['rank_metrics']['2']['heldout']['retained_energy_fraction'],'global_scale':controller['global_scale'],'feature_component_shape':list(basis.shape),
        'component_gram_matrix':component_gram.tolist(),'intervention_site':'after zero-based Block 1','token_scope':'condition-prefix tokens only','fm_calls':20,'observed_window_frames':32}
    write_json(out/'pca_summary.json',pca_summary)
    release_split={**split,'frozen_before_new_pca':True,'pca_fit_pair_ids':sorted(train),'final_evaluation_pair_ids':sorted(heldout),'rank_selected_from_fit_only':2,'intervention_site_selection':'deepest maximum full-matched E3 recovery plateau on fit split only'};write_json(out/'split_manifest.json',release_split)

    coordinate=[{**row,'delta_gravity':None} for row in controller['coordinate_predictions']];write_csv(out/'coordinate_predictions.csv',coordinate)
    coordinate_summary={'controller':controller['controller'],'feature_order':controller['feature_order'],'fit_pairs':64,'heldout_pairs':64,'pca_rank':2,'global_scale':controller['global_scale'],'models':controller['models']};write_json(out/'coordinate_summary.json',coordinate_summary)

    f1_by_key={}
    for path in sorted(a.f1_root.glob('**/rows.json')):
        for row in read_json(path):f1_by_key[(row['history_regime'],row['pair_id'],row['hue_index'])]=row
    f1=list(f1_by_key.values())
    assert len(f1)==1408 and len({(r['history_regime'],r['pair_id'],r['hue_index']) for r in f1})==1408
    behavior=[];registry=[]
    for r in f1:
        name=Path(r['output_video_path']).name;candidates=sorted(a.f1_root.glob(f'**/videos/{name}'));assert candidates,(name,)
        source=next((path for path in candidates if sha(path)==r['output_video_sha256']),candidates[0]);dest=out/'videos/behavior'/source.name;shutil.copy2(source,dest)
        row={**r,'output_video_path':str(dest.relative_to(out)).replace('\\','/'),'output_video_sha256':sha(dest)};behavior.append(row)
        registry.append({'scientific_role':'F1 evidence-dependent behavior','pair_id':r['pair_id'],'condition':f"{r['history_regime']}_hue_{r['hue_index']}",'path':row['output_video_path'],'sha256':row['output_video_sha256'],'generation_seed':r['generation_seed'],'fm_calls':20})
    behavior_fields=['task','model_seed','checkpoint_step','history_regime','trajectory_id','pair_id','generation_seed','hue_u','input_r','input_g','input_b','true_gravity_band','gravity_true','gravity_hat_e3','gravity_hat_e0','fit_rmse','valid','invalid_reasons','outcome','future_r','future_g','future_b','future_color_coordinate','detector_coverage','max_missing_run','max_adjacent_y_jump','output_video_path','output_video_sha256']
    write_csv(out/'behavior_rollouts.csv',behavior,behavior_fields);write_json(out/'behavior_rollouts.json',behavior)
    behavior_summary={'rows':len(behavior),'physical_histories':len({r['pair_id'] for r in behavior}),'hues':11,'regimes':['Short','Long'],'valid_count':int(sum(r['valid'] for r in behavior)),'invalid_count':int(sum(not r['valid'] for r in behavior)),'valid_rate':float(np.mean([r['valid'] for r in behavior])),
        'descriptive_only':True,'raw_points_and_per_hue_medians':'no inferential uncertainty interval'};write_json(out/'behavior_summary.json',behavior_summary)

    strict=[]
    for item in pairs:
        pid=item['pair_id'];ma=meta[(pid,'aligned')];mc=meta[(pid,'conflict')];ea=evals[(pid,'aligned')];ec=evals[(pid,'conflict')];b=boundary_by[pid]
        strict.append({'receiver_id':pid,'pair_id':pid,'direction':item['gravity_interval'],'split':'train' if pid in train else 'heldout','history_regime':'hist32',
            'y_boundary':float(ma['y_boundary']),'vy_boundary':float(ma['vy_boundary']),'x_boundary':float(ma['x_boundary']),'vx_boundary':float(ma['vx_boundary']),'gravity_true':float(item['gravity_true']),
            'aligned_color':ma['color_label'],'conflict_color':mc['color_label'],'gravity_hat_aligned':ea['g_E3'],'gravity_hat_conflict':ec['g_E3'],'recovery_denominator':ea['g_E3']-ec['g_E3'],
            'aligned_valid':ea['valid_track'],'conflict_valid':ec['valid_track'],'aligned_video_path':f"source:{ma['video']}",'conflict_video_path':f"source:{mc['video']}",
            'aligned_input_video_sha256':b['aligned_input_video_sha256'],'conflict_input_video_sha256':b['conflict_input_video_sha256'],'generation_seed':int(ma['base_seed'])+17000000})
    write_csv(out/'strict_bank.csv',strict);write_json(out/'strict_bank.json',strict)

    controller_rows=[]
    for path in sorted((work/'controller_extracted/controller_rollout').glob('shard*/outcomes.json')):controller_rows+=read_json(path)
    assert len(controller_rows)==128
    controller_by={(r['receiver_id'],r['condition']):r for r in controller_rows}
    recovery_all=read_json(work/'recovery-summary/outcomes_merged.json');recovery_by={(r['pair_id'],r['rank']):r for r in recovery_all}
    decoded=[];recovery_video_root=out/'videos/recovery'
    for pid in sorted(heldout):
        m=meta[(pid,'conflict')];gc=float(evals[(pid,'conflict')]['g_E3']);ga=float(evals[(pid,'aligned')]['g_E3']);denom=ga-gc;idx=int(np.where(np.asarray(packed['pair_ids'],dtype=str)==pid)[0][0])
        conditions=[('natural_conflict',find_controller_frame(work/'controller_extracted/controller_rollout',pid,'natural_conflict'),controller_by[(pid,'natural_conflict')],0.,0.),
                    ('full_matched_edit',find_frame(work/'recovery_extracted','full_delta',pid),recovery_by[(pid,'full_delta')],1.,float(np.sqrt(packed['squared_norms'][idx]))),
                    ('top2_oracle_projection',find_frame(work/'recovery_extracted','2',pid),recovery_by[(pid,'2')],controller['global_scale'],float(np.linalg.norm(packed['scores'][idx,:2])*controller['global_scale'])),
                    ('fit_only_controller',find_controller_frame(work/'controller_extracted/controller_rollout',pid,'fit_only_controller'),controller_by[(pid,'fit_only_controller')],controller['global_scale'],float(controller_by[(pid,'fit_only_controller')]['activation_delta_l2']))]
        for condition,npz,source,scale,norm in conditions:
            dest=recovery_video_root/f'{pid}__{condition}.mp4';frames=to_mp4(npz,dest);audit=audit_frames(frames,m);g=audit['gravity_hat_e3'];R=0. if condition=='natural_conflict' else (g-gc)/denom
            row={'receiver_id':pid,'pair_id':pid,'condition':condition,'direction':pair_map[pid]['gravity_interval'],'gravity_hat_conflict':gc,'gravity_hat_aligned':ga,'gravity_hat_edited':g,
                'recovery_denominator':denom,'normalized_recovery_g':R,'valid':audit['valid'],'invalid_reasons':audit['invalid_reasons'],'outcome':'E3_pass' if abs(g-float(m['gravity']))<.002 else 'E3_fail',
                'future_color_coordinate':audit['future_color_coordinate'],'future_r':audit['future_r'],'future_g':audit['future_g'],'future_b':audit['future_b'],'fit_rmse':audit['fit_rmse'],
                'gravity_hat_e0':audit['gravity_hat_e0'],'detector_coverage':audit['detector_coverage'],'max_missing_run':audit['max_missing_run'],'max_adjacent_y_jump':audit['max_adjacent_y_jump'],
                'intervention_site':'none' if condition=='natural_conflict' else 'after_block_1','fm_calls_observed':20,'fm_calls_expected':20,
                'condition_prefix_tokens_edited':0 if condition=='natural_conflict' else 1088,'target_suffix_tokens_edited':0,'intervention_scale':scale,'activation_delta_l2':norm,
                'generation_seed':int(m['base_seed'])+17000000,'output_video_path':str(dest.relative_to(out)).replace('\\','/'),'output_video_sha256':sha(dest)}
            decoded.append(row);registry.append({'scientific_role':'F3 heldout decoded recovery','pair_id':pid,'condition':condition,'path':row['output_video_path'],'sha256':row['output_video_sha256'],'generation_seed':row['generation_seed'],'fm_calls':20})
    write_csv(out/'decoded_recovery.csv',decoded)
    condition_summary={}
    for condition in ('natural_conflict','full_matched_edit','top2_oracle_projection','fit_only_controller'):
        rows=[r for r in decoded if r['condition']==condition];condition_summary[condition]={'n':64,'valid':sum(r['valid'] for r in rows),'E3_correct':sum(r['outcome']=='E3_pass' for r in rows),
            'E3_accuracy':float(np.mean([r['outcome']=='E3_pass' for r in rows])),'near_full_R_count':sum(r['valid'] and .9<=r['normalized_recovery_g']<=1.1 for r in rows),
            'near_full_R_rate':float(np.mean([r['valid'] and .9<=r['normalized_recovery_g']<=1.1 for r in rows])),'mean_R':float(np.mean([r['normalized_recovery_g'] for r in rows]))}
    decoded_summary={'heldout_receivers':64,'rank':2,'conditions':condition_summary,'same_rank_oracle_and_controller':True,'same_global_scale':controller['global_scale'],'invalid_in_denominator':True};write_json(out/'decoded_recovery_summary.json',decoded_summary)

    layer_source=[]
    for path in sorted((work/'delivery_inputs/layer').glob('outcomes_shard*.json')):layer_source+=read_json(path)
    replacements=[r for r in layer_source if r['condition']=='replacement' and r['pair_id'] in heldout];fit_replacements=[r for r in layer_source if r['condition']=='replacement' and r['pair_id'] in train]
    assert len(replacements)==len(fit_replacements)==64*30
    def layer_rows(source,split_label):
        rows=[]
        for r in source:
            pid=r['pair_id'];gc=float(evals[(pid,'conflict')]['g_E3']);ga=float(evals[(pid,'aligned')]['g_E3']);R=(r['gravity_hat']-gc)/(ga-gc) if r['gravity_hat'] is not None else None
            rows.append({'receiver_id':pid,'history_regime':'hist32','direction':r['true_band'],'split':split_label,'site_index':r['block'],'site_label':f"after_block_{r['block']}",
                'gravity_hat_conflict':gc,'gravity_hat_aligned':ga,'gravity_hat_edited':r['gravity_hat'],'normalized_recovery_g':R,'valid':r['valid'],'invalid_reasons':'' if r['valid'] else 'detected_frames_below_58',
                'writable_indicator':bool(r['valid'] and R is not None and .9<=R<=1.1),'common_strict':bool(evals[(pid,'conflict')]['valid_track'] and evals[(pid,'aligned')]['valid_track'] and r['valid'])})
        return rows
    layers=layer_rows(replacements,'heldout');fit_layers=layer_rows(fit_replacements,'train');write_csv(out/'layer_scan.csv',layers)
    train_site=[]
    for site in range(30):
        rows=[r for r in fit_layers if r['site_index']==site];train_site.append({'site':site,'E3_accuracy':float(np.mean([abs(r['gravity_hat_edited']-float(meta[(r['receiver_id'],'conflict')]['gravity']))<.002 for r in rows]))})
    best=max(r['E3_accuracy'] for r in train_site);selected=max(r['site'] for r in train_site if r['E3_accuracy']==best);assert selected==1
    by_direction={}
    for band in ('low','high'):
        stats=[]
        for site in range(30):
            rows=[r for r in layers if r['direction']==band and r['site_index']==site];values=[r['normalized_recovery_g'] for r in rows if r['valid'] and r['normalized_recovery_g'] is not None]
            stats.append({'site':site,'n':len(rows),'valid_rate':float(np.mean([r['valid'] for r in rows])),'mean_R':float(np.mean(values)),'median_R':float(np.median(values)),'q25_R':float(np.quantile(values,.25)),'q75_R':float(np.quantile(values,.75)),'W_g':float(np.mean([r['writable_indicator'] for r in rows]))})
        w=np.asarray([r['W_g'] for r in stats]);above=np.flatnonzero(w>=.5);high_sites=np.flatnonzero(w>=.9);low_after=np.flatnonzero((np.arange(30)>high_sites.max())&(w<=.1)) if len(high_sites) else np.asarray([])
        by_direction[band]={'sites':stats,'D_g_effective_writable_depth':float(w.sum()),'D_g_definition':'sum across sites of receiver-level writable proportion W_g',
            'L50':int(above.max()) if len(above) else None,'L50_definition':'deepest after-block site with W_g >= 0.5','closure_interval':[int(above.min()),int(above.max())] if len(above) else None,
            'transition_width':int(low_after.min()-high_sites.max()) if len(high_sites) and len(low_after) else None,'maximum_adjacent_drop':float(np.max(w[:-1]-w[1:]))}
    layer_summary={'site_convention':'after zero-based DiT blocks; 30 sites 0..29','fit_only_site_selection':train_site,'selected_site':selected,'selected_site_rule':'deepest site on maximum fit-split E3 accuracy plateau',
        'heldout_pairs':64,'by_direction':by_direction};write_json(out/'layer_summary.json',layer_summary)

    evaluator=[]
    for pid in sorted(train|heldout):
        for condition in ('aligned','conflict'):
            r=evals[(pid,condition)];evaluator.append({'source':'natural_model','receiver_id':pid,'condition':condition,'gravity_true':r['gravity_true'],'gravity_hat_e3':r['g_E3'],'gravity_hat_e0':r['g_E0_strict'],'fit_rmse':r['y_rmse_E3'],'valid':r['valid_track'],'invalid_reasons':'','detected_frames':r['detected_frames'],'detector_coverage':r['detected_frames']/64,'max_missing_run':'','max_adjacent_y_jump':'','future_r':'','future_g':'','future_b':'','future_color_coordinate':r['detected_colour']})
    for r in behavior:evaluator.append({'source':'F1','receiver_id':r['pair_id'],'condition':f"{r['history_regime']}_hue_{r['hue_index']}",'gravity_true':r['gravity_true'],'gravity_hat_e3':r['gravity_hat_e3'],'gravity_hat_e0':r['gravity_hat_e0'],'fit_rmse':r['fit_rmse'],'valid':r['valid'],'invalid_reasons':r['invalid_reasons'],'detected_frames':r['detected_frames'],'detector_coverage':r['detector_coverage'],'max_missing_run':r['max_missing_run'],'max_adjacent_y_jump':r['max_adjacent_y_jump'],'future_r':r['future_r'],'future_g':r['future_g'],'future_b':r['future_b'],'future_color_coordinate':r['future_color_coordinate']})
    for r in decoded:evaluator.append({'source':'F3','receiver_id':r['receiver_id'],'condition':r['condition'],'gravity_true':float(meta[(r['receiver_id'],'conflict')]['gravity']),'gravity_hat_e3':r['gravity_hat_edited'],'gravity_hat_e0':r['gravity_hat_e0'],'fit_rmse':r['fit_rmse'],'valid':r['valid'],'invalid_reasons':r['invalid_reasons'],'detected_frames':r['detector_coverage']*64,'detector_coverage':r['detector_coverage'],'max_missing_run':r['max_missing_run'],'max_adjacent_y_jump':r['max_adjacent_y_jump'],'future_r':r['future_r'],'future_g':r['future_g'],'future_b':r['future_b'],'future_color_coordinate':r['future_color_coordinate']})
    for r in layers:evaluator.append({'source':'F4_source','receiver_id':r['receiver_id'],'condition':r['site_label'],'gravity_true':float(meta[(r['receiver_id'],'conflict')]['gravity']),'gravity_hat_e3':r['gravity_hat_edited'],'gravity_hat_e0':'','fit_rmse':'','valid':r['valid'],'invalid_reasons':r['invalid_reasons'],'detected_frames':'','detector_coverage':'','max_missing_run':'','max_adjacent_y_jump':'','future_r':'','future_g':'','future_b':'','future_color_coordinate':''})
    write_csv(out/'evaluator_audit.csv',evaluator);write_json(out/'evaluator_audit.json',evaluator)
    evaluator_contract={'task':'free_fall','primary_evaluator':'E3 free-state gravity fit with position and velocity offsets','secondary_evaluator':'E0 fixed-boundary-state gravity fit','E3_threshold_abs_g_error':.002,'E0_threshold_abs_g_error':.002,
        'trajectory_validity':'at least 58 of 64 future frames detected','invalid_in_denominator':True,'time_axis':'tau=(future_frame_index+1)/20 s','same_contract_conditions':['natural','full matched','top-2 oracle','fit-only controller','layer scan'],
        'known_source_gap':'Archived F4 layer-scan output did not retain fit RMSE, coverage, jump, or future RGB fields; these are blank in evaluator_audit.csv.'};write_json(out/'evaluator_contract.json',evaluator_contract)

    # F1
    fig,axes=plt.subplots(2,2,figsize=(3862/600,2419/600),sharex=True,sharey=True,layout='constrained')
    hues=np.arange(11)/10
    for ri,band in enumerate(('low','high')):
        for ci,regime in enumerate(('Short','Long')):
            ax=axes[ri,ci];rows=[r for r in behavior if r['true_gravity_band']==band and r['history_regime']==regime]
            for r in rows:
                color=np.clip(np.asarray([r['future_r'],r['future_g'],r['future_b']])/255,0,1) if r['valid'] else INVALID
                ax.scatter(r['hue_u'],r['gravity_hat_e3'],s=5,c=[color],alpha=.55,marker='o' if r['valid'] else 'x',linewidths=.35)
            med=[np.median([r['gravity_hat_e3'] for r in rows if r['hue_index']==j and r['valid']]) for j in range(11)]
            ax.plot(hues,med,c=INK,lw=1,marker='o',ms=2);ax.axhspan(.003,.025,color='#F3D6D1',alpha=.45);ax.axhspan(.025,.045,color='#F8F6F1',alpha=.55);ax.axhspan(.045,.067,color='#D8E1F0',alpha=.45)
            ax.set_title(regime);ax.text(.02,.95,f"invalid: {sum(not r['valid'] for r in rows)}/{len(rows)}",transform=ax.transAxes,va='top',fontsize=6.2)
            if ci==0:ax.set_ylabel(('Low-gravity history\n' if band=='low' else 'High-gravity history\n')+'Decoded gravity $\\hat g$')
            if ri==1:ax.set_xlabel('Input cue $u$ (red $\\rightarrow$ blue)')
    fig.suptitle('Free fall: appearance cue and observed history jointly shape decoded gravity',y=.995);fig.text(.5,.005,'Raw rollouts and per-hue medians are descriptive; point color is measured future RGB.',ha='center',fontsize=6.2,color=SECONDARY);save_figure(fig,out/'figures','freefall_behavior')
    # F2
    fig,axes=plt.subplots(1,2,figsize=(3844/600,1831/600),layout='constrained')
    for comp,ax in enumerate(axes,1):
        rows=[r for r in coordinate if r['coordinate']==comp]
        for split_name,marker,alpha in [('train','o',.4),('heldout','s',.9)]:
            for band,color in [('low',LOW),('high',HIGH)]:
                pts=[r for r in rows if r['split']==split_name and r['direction']==band];ax.scatter([r['observed_score'] for r in pts],[r['predicted_score'] for r in pts],s=12,marker=marker,c=color,alpha=alpha,label=f'{band} {split_name}')
        values=[r['observed_score'] for r in rows]+[r['predicted_score'] for r in rows];lo,hi=min(values),max(values);ax.plot([lo,hi],[lo,hi],c=INK,lw=.7);ax.set(xlabel=f'Observed $z_{comp}$',ylabel=f'Fit-only predicted $z_{comp}$',title=f'Causal coordinate $z_{comp}$')
        if comp==1:ax.text(.03,.96,'$\\hat z^{(r)}(g)=\\beta_0^{(r)}+\\beta_g^{(r)}g$',transform=ax.transAxes,va='top',fontsize=6.5)
        else:ax.legend(frameon=False,fontsize=5.8,ncol=2)
    fig.suptitle(f"Free fall: held-out coordinate prediction (joint $R^2$: low {controller['models']['low']['heldout_joint_R2']:.3f}, high {controller['models']['high']['heldout_joint_R2']:.3f})",y=.995);save_figure(fig,out/'figures','freefall_state_geometry')
    # F3
    fig=plt.figure(figsize=(3887/600,1983/600),layout='constrained');gs=fig.add_gridspec(2,2,width_ratios=[2.4,1]);ax=fig.add_subplot(gs[:,0]);labels=['Natural conflict','Full matched','Top-2 oracle','Fit-only controller'];conds=['natural_conflict','full_matched_edit','top2_oracle_projection','fit_only_controller']
    rng=np.random.default_rng(3407)
    for x,(label,cond) in enumerate(zip(labels,conds)):
        rows=[r for r in decoded if r['condition']==cond]
        for band,color,offset in [('low',LOW,-.08),('high',HIGH,.08)]:
            pts=[r for r in rows if r['direction']==band];jitter=rng.uniform(-.035,.035,len(pts));ax.scatter(x+offset+jitter,[r['normalized_recovery_g'] for r in pts],s=8,c=color,alpha=.72)
        near=sum(.9<=r['normalized_recovery_g']<=1.1 and r['valid'] for r in rows);valid=sum(r['valid'] for r in rows);ax.text(x,1.17,f'{near}/64 near-full\n{valid}/64 valid',ha='center',va='bottom',fontsize=5.6)
    ax.axhspan(.9,1.1,color='#D8ECE8',alpha=.7);ax.axhline(1,c=TEAL,lw=.8);ax.axhline(0,c=INK,lw=.6);ax.set(xticks=range(4),xticklabels=labels,ylabel='Normalized gravity recovery $R^g$',ylim=(-.12,1.32));ax.tick_params(axis='x',rotation=12)
    rep=min([r for r in decoded if r['condition']=='fit_only_controller' and r['direction']=='low'],key=lambda r:abs(r['normalized_recovery_g']-1))['receiver_id']
    for rowi,condition in enumerate(('natural_conflict','fit_only_controller')):
        path=recovery_video_root/f'{rep}__{condition}.mp4';frames=np.asarray(imageio.mimread(path));sub=gs[rowi,1].subgridspec(1,3,wspace=.03)
        for j,index in enumerate((0,31,63)):
            iax=fig.add_subplot(sub[0,j]);iax.imshow(frames[index]);iax.axis('off');
            if j==0:iax.set_title('Natural conflict' if rowi==0 else 'Fit-only edit',loc='left',fontsize=6.5)
    fig.suptitle(f'Free fall: donor-free held-out recovery at rank 2 (representative {rep})',y=.995);save_figure(fig,out/'figures','freefall_decoded_recovery')
    # F4
    fig,axes=plt.subplots(1,2,figsize=(3848/600,1768/600),sharey=True,layout='constrained')
    for ax,band,color in zip(axes,('low','high'),(LOW,HIGH)):
        band_rows=[r for r in layers if r['direction']==band];by=defaultdict(list)
        for r in band_rows:by[r['receiver_id']].append(r)
        for pid,rows in by.items():rows=sorted(rows,key=lambda r:r['site_index']);ax.plot([r['site_index'] for r in rows],[r['normalized_recovery_g'] for r in rows],c=color,alpha=.08,lw=.35)
        stats=by_direction[band]['sites'];x=np.arange(30);median=np.asarray([r['median_R'] for r in stats]);q1=np.asarray([r['q25_R'] for r in stats]);q3=np.asarray([r['q75_R'] for r in stats]);ax.fill_between(x,q1,q3,color=color,alpha=.18);ax.plot(x,median,c=color,lw=1.2)
        interval=by_direction[band]['closure_interval'];
        if interval:ax.axvspan(interval[0]-.4,interval[1]+.4,color=AMBER,alpha=.15)
        ax.axhspan(.9,1.1,color='#D8ECE8',alpha=.45);ax.axhline(1,c=TEAL,lw=.7);ax.set(xlabel='Residual site (after zero-based block)',title=f'Target {band} | $L_{{50}}$={by_direction[band]["L50"]}',xlim=(-.5,29.5))
    axes[0].set_ylabel('Normalized gravity recovery $R^g$');fig.suptitle('Free fall: direct writeability closes over an early depth interval',y=.995);save_figure(fig,out/'figures','freefall_writeability')

    # Canonical representative videos and posters.
    canonical={'natural_conflict':recovery_video_root/f'{rep}__natural_conflict.mp4','state_edit':recovery_video_root/f'{rep}__fit_only_controller.mp4','top2_oracle':recovery_video_root/f'{rep}__top2_oracle_projection.mp4','full_matched_edit':recovery_video_root/f'{rep}__full_matched_edit.mp4'}
    for name,source in canonical.items():shutil.copy2(source,out/'videos'/f'{name}.mp4')
    aligned_dest=out/'videos/aligned_reference.mp4';aligned_frames=to_mp4(a.aligned_reference,aligned_dest)
    for name,path in {**canonical,'aligned_reference':aligned_dest}.items():
        actual=out/'videos'/f'{name}.mp4' if name!='aligned_reference' else aligned_dest;frames=np.asarray(imageio.mimread(actual));imageio.imwrite(out/'videos/posters'/f'{name}.png',frames[len(frames)//2])
        registry.append({'scientific_role':'representative F3 strip','pair_id':rep,'condition':name,'path':str(actual.relative_to(out)).replace('\\','/'),'sha256':sha(actual),'generation_seed':int(meta[(rep,'conflict')]['base_seed'])+17000000,'fm_calls':20})
    with (out/'video_registry.jsonl').open('w',encoding='utf-8') as f:
        for r in registry:f.write(json.dumps(r)+'\n')

    for source in [work/'fit_train_only_pca.py',work/'fit_directional_trainonly.py',work/'rollout_directional_trainonly.py',work/'freefall_hue_sweep_delivery.py',work/'build_boundary_audit.py',work/'export_trainonly_basis.py',work/'build_panel_kit.py',work/'audit_freefall_contract.py',work/'validate_figures.py',work/'finalize_manifest.py']:
        shutil.copy2(source,out/'scripts'/source.name)
    for name,description in [('build_freefall_behavior.py','Rebuild F1 behavior figure'),('build_freefall_state_geometry.py','Rebuild F2 coordinate figure'),('build_freefall_decoded_recovery.py','Rebuild F3 recovery figure'),('build_freefall_writeability.py','Rebuild F4 writeability figure')]:
        (out/'scripts'/name).write_text(f'''#!/usr/bin/env python3\n"""{description}.\n\nThe release is built atomically by build_panel_kit.py so all tables, videos,\nfigures, and hashes remain synchronized. See REBUILD.md for the exact command.\n"""\nfrom pathlib import Path\nprint((Path(__file__).resolve().parents[1] / "REBUILD.md").read_text(encoding="utf-8"))\n''',encoding='utf-8')
    methods=f'''# Free Fall PCA and Controller Methods\n\nThe PCA basis is fitted from 64 training pairs only. The 64 held-out pairs never enter the basis, rank selection, scale, controller fit, or intervention-site selection. Rank 2 is the smallest rank retaining at least 99% of training residual energy.\n\nThe official donor-free controller is direction-specific:\n\n```text\nz_hat^(r)(g_target) = beta_0^(r) + beta_g^(r) g_target,  r in {{low, high}}\n```\n\nIt predicts two coordinates using target direction and target gravity only. Both oracle and controller use the same train-only PCA basis, global fit-only scale `{controller['global_scale']:.9f}`, after-Block-1 site, condition-prefix token scope, 20 flow-matching calls, generation seed, and evaluator.\n\nHeld-out coordinate joint R2 is `{controller['models']['low']['heldout_joint_R2']:.6f}` for low and `{controller['models']['high']['heldout_joint_R2']:.6f}` for high. The top-2 oracle has E3 `{condition_summary['top2_oracle_projection']['E3_accuracy']:.4%}` and the fit-only controller has E3 `{condition_summary['fit_only_controller']['E3_accuracy']:.4%}` on 64 held-out receivers.\n''';(out/'CONTROLLER_METHODS.md').write_text(methods,encoding='utf-8')
    qa=f'''# QA\n\n- Frozen bank: 128 pairs. PCA/controller fit: 64. Final held-out evaluation: 64. Low/high counts are 32/32 in each split.\n- Pair IDs and base seeds are disjoint across fit and held-out. One legacy crossing at base seed 50208 was repaired before PCA; see `split_manifest.json`.\n- Boundary audit: 128/128 pass at tolerance 1e-12; maximum recorded boundary error is 0. Renderer nuisance hashes and generation seeds match within every pair.\n- PCA: rank 2 selected from fit energy only ({pca_summary['train_retained_energy_fraction']:.6%}); held-out energy ({pca_summary['heldout_retained_energy_fraction']:.6%}) is diagnostic only. Exported feature components are orthonormal.\n- Intervention site: after zero-based Block 1, selected as the deepest maximum-E3 site on the 64-pair fit split.\n- F1: 1,408/1,408 futures regenerated and valid; measured future RGB, coverage, gap/jump diagnostics, MP4 path, and SHA256 are recorded. Raw points and medians are descriptive.\n- F3: all four conditions contain the same 64 held-out receivers. All 256 videos are valid. Top-2 oracle E3 is {condition_summary['top2_oracle_projection']['E3_correct']}/64; fit-only controller E3 is {condition_summary['fit_only_controller']['E3_correct']}/64. Invalid outputs remain in denominators.\n- F4: `layer_scan.csv` has 1,920 receiver-level rows (64 held-out x 30 sites). It reuses the archived full-matched layer scan. The archive did not retain RMSE, coverage, jump, or RGB diagnostics for each layer edit; those fields are blank in `evaluator_audit.csv`.\n- Figures: four separate figures exported as editable SVG, PDF, and 600-dpi PNG using the supplied Pendulum palette and canvas sizes. PDF render and font audits are recorded in `figure_QA.json`.\n- Rebuild: scripts, input hashes, producer names, file sizes, and SHA256 values are recorded in `manifest.json`. Model repository HEAD was 8386c6f0f7d23f38d1e74157da61c0ba47c246db with a dirty working tree; delivery scripts are therefore pinned by SHA256.\n- Scientific scope: the 128-pair bank was selected using baseline aligned/conflict E3 behavior and had appeared in prior analyses. Held-out isolation is correct for the rebuilt PCA/controller pipeline, but this is not a fresh unseen population. The F4 source scan predates the repaired split; site selection is recomputed using fit IDs only.\n''';(out/'QA.md').write_text(qa,encoding='utf-8')
    rebuild=f'''# Deterministic rebuild\n\nRun `scripts/build_panel_kit.py` with the retained source bundle, regenerated F1 directory, exported train-only basis, and aligned-reference NPZ. The exact invocation used for this release was:\n\n```text\npython build_panel_kit.py --work "{work.resolve()}" --old-kit "{a.old_kit.resolve()}" --f1-root "{a.f1_root.resolve()}" --basis "{a.basis.resolve()}" --aligned-reference "{a.aligned_reference.resolve()}" --out "{out.resolve()}"\n```\n\nThe source model repository had uncommitted changes. Reproducibility therefore depends on the SHA256-pinned scripts in this package and the source artifacts named in the manifest. After rebuilding, run `python scripts/audit_freefall_contract.py <kit-path>`.\n''';(out/'REBUILD.md').write_text(rebuild,encoding='utf-8')
    readme=f'''# Free fall panel kit: train-only PCA, rank 2\n\nThis package supersedes the earlier panel kit for PCA-dependent claims. It uses 64 pairs to fit an uncentered Block-1 PCA basis and reserves 64 disjoint pairs for final evaluation. The official controller is direction-specific `[1,g_target]` and never reads a held-out aligned activation or generated gravity.\n\nHeadline held-out results:\n\n- Top-2 oracle projection: {condition_summary['top2_oracle_projection']['E3_correct']}/64 E3 correct; {condition_summary['top2_oracle_projection']['near_full_R_count']}/64 with `0.9 <= R <= 1.1`.\n- Fit-only controller: {condition_summary['fit_only_controller']['E3_correct']}/64 E3 correct; {condition_summary['fit_only_controller']['near_full_R_count']}/64 with `0.9 <= R <= 1.1`; mean R {condition_summary['fit_only_controller']['mean_R']:.6f}.\n- Full matched edit: {condition_summary['full_matched_edit']['E3_correct']}/64 E3 correct.\n\nThe four paper figures are separate under `figures/`. Machine-readable tables, full videos, representative strips, scripts, hashes, and audit records are included. Read `QA.md` before using the figures in a paper; the archived layer scan has narrower evaluator diagnostics than the newly generated F1 and F3 videos.\n''';(out/'README.md').write_text(readme,encoding='utf-8')
    manifest={'schema_version':'2.0','system':'free_fall','paper_label':'Free fall','status':'train_only_pca2_directional_controller_complete','pca_fit_pairs':64,'heldout_pairs':64,'files':[]}
    roles={'behavior_rollouts.csv':'F1 raw rollout table','coordinate_predictions.csv':'F2 coordinate predictions','decoded_recovery.csv':'F3 heldout recovery table','layer_scan.csv':'F4 heldout layer scan','boundary_match_audit.csv':'boundary gate','evaluator_audit.csv':'frozen evaluator audit'}
    producers={'behavior_rollouts.csv':'freefall_hue_sweep_delivery.py','coordinate_predictions.csv':'fit_directional_trainonly.py','decoded_recovery.csv':'build_panel_kit.py','layer_scan.csv':'build_panel_kit.py','boundary_match_audit.csv':'build_boundary_audit.py'}
    for path in sorted(p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'):
        rel=str(path.relative_to(out)).replace('\\','/');manifest['files'].append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path),'producer':producers.get(rel,'build_panel_kit.py'),'source_table':rel if rel.endswith(('.csv','.json')) else None,'scientific_role':roles.get(rel,'supporting release artifact')})
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'out':str(out),'manifest_files':len(manifest['files']),'behavior_rows':len(behavior),'decoded_rows':len(decoded),'layer_rows':len(layers),'representative':rep,'controller_summary':condition_summary['fit_only_controller']},indent=2))


if __name__=='__main__':main()
