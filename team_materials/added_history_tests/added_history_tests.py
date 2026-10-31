import json, sys, os, hashlib, re, warnings
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
display = lambda *a, **k: None
NB = json.load(open(sys.argv[1])); FIX = sys.argv[2]; ROOT = Path(sys.argv[3]); OUT = Path(sys.argv[4])
CONFIG = {'seed':20261019,'logistic_C':1.,'logistic_max_iter':500,'validation_start':'2026-10-03T00:00:00Z','test_start':'2026-10-04T00:00:00Z'}
# --- data load (as cell 4) ---
d = pd.read_parquet(ROOT/'data/wallet_market_candidates_full_week.parquet'); metadata = pd.read_csv(ROOT/'data/weekly_market_metadata_complete.csv')
assert len(d)==96184 and d.conditional_profit_label.notna().sum()==89736
for c in ['prediction_time','feature_week_history_first_time','feature_week_history_latest_time']: d[c]=pd.to_datetime(d[c],utc=True)
metadata['resolved_at']=pd.to_datetime(metadata.resolved_at,utc=True,format='mixed',errors='coerce')
mapping_evidence = pd.read_csv(ROOT/'data/settlement_mapping_audit.csv')
# --- 24 h features (saved output of cell 6, instead of recomputing from raw trades) ---
f = pd.read_parquet(FIX); f['prediction_time']=pd.to_datetime(f.prediction_time,utc=True)
d = d.merge(f.drop(columns=['prediction_time']), on=['wallet','market'], how='left', validate='one_to_one')
assert len(d)==96184
for c in ['fill_rows','markets','buy_dollars','avg_buy_ticket']:
    if 'feature_fixed24h_log_'+c not in d: d['feature_fixed24h_log_'+c]=np.log1p(d['feature_fixed24h_'+c])
# --- cells 10, 12, 14 unchanged ---
src = lambda i: ''.join(NB['cells'][i]['source'])
for i in [10, 12, 14]: exec(src(i), globals())
ref = pd.read_csv(sys.argv[5])
m = tests.merge(ref, on=['lookback','hypothesis'], suffixes=('','_ref'))
print('reproduced 8 frozen tests, max |p diff|:', float((m.p_raw-m.p_raw_ref).abs().max()), ' rows:', len(train))
# --- added: the two 5-feature extras, same test, one at a time, with their missing flags as controls ---
extra=[]
for period in ['expanding','fixed24h']:
    src_col = 'feature_week_history_' if period=='expanding' else 'feature_fixed24h_'
    for name, col in [('buy_hhi', src_col+'buy_market_hhi'), ('log_avg_buy_ticket', src_col+'log_avg_buy_ticket')]:
        x = train[col]
        full_inference[period+'__'+name] = x
        full_inference[period+'__'+name+'__missing'] = x.isna().astype(float)
        extra.append((period, name, int(x.isna().sum())))
common_pre=TrainPreprocessor().fit(full_inference)
common_matrix=common_pre.transform(full_inference); common_names=list(full_inference)
Xwithin=within(common_matrix,market)
rows=[]
for period,name,nmiss in extra:
    term=period+'__'+name; flag=term+'__missing'
    sel=list(base)+[term]+([flag] if nmiss>0 else [])
    fitted=nested_fit(sel, period+'_only_'+name)
    t=wald_test(fitted,[term]); co=fitted[0].set_index('term').loc[term]
    rows.append({'lookback':period,'hypothesis':'single_'+name,'p_raw':t['p_raw'],'coefficient':co.coefficient,'missing_rows':nmiss,'status':t['test_status']})
# same 3 baseline tests again with the extended matrix: must not change
chk = wald_test(nested_fit(list(base)+['expanding__log_buy_dollars'],'chk'),['expanding__log_buy_dollars'])['p_raw']
print('check unchanged p (expanding buy $):', chk)
ex=pd.DataFrame(rows)
ex['p_holm_4_added']=multipletests(ex.p_raw,method='holm')[1]
allp=pd.concat([tests[['lookback','hypothesis','p_raw']],ex[['lookback','hypothesis','p_raw']]],ignore_index=True)
allp['p_holm_12']=multipletests(allp.p_raw,method='holm')[1]
print(ex.to_string()); print(allp.to_string())
ex.to_csv(OUT/'added_tests_hhi_avg_buy.csv',index=False); allp.to_csv(OUT/'all_12_tests_holm.csv',index=False)
