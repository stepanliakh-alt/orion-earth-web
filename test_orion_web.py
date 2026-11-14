"""Offline tests of ORION Earth calculation and NASA JSON parser.
Synthetic fixtures exist only in these tests; the app never displays them.
Run: python tests/test_core.py
"""
from __future__ import annotations
import ast
import calendar
import hashlib
import html
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import kendalltau, theilslopes
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
source = ROOT.joinpath('app.py').read_text('utf-8')
mod = ast.parse(source)
names = {
    'longest_run_of_dry_days','nasa_request_params','nasa_cache_path',
    'validated_parameters','read_cache_record','save_cache_record','fetch_nasa_power',
    'count_dry_to_rain_events','aggregate_annual','calc_trend','label_trend',
    'detect_anomalies','analyze_site','implications','evidence_comparison','trend_plot',
    'aggregate_monthly','longest_boolean_run','segment_indicators',
    'aggregate_seasonal','fdr_bh','segmented_trends','contiguous_events',
    'render_offline_showcase',
}
functions = []
for node in mod.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        node.decorator_list = []
        functions.append(node)
assert {n.name for n in functions} == names

env = dict(globals())
env.update({
    'PARAMETERS': ['PRECTOTCORR','T2M','T2M_MAX','T2M_MIN'],
    'NASA_URL': 'https://power.larc.nasa.gov/api/temporal/daily/point',
    'METRICS': {
        name:(name,'unit','unit/10years') for name in [
            'rain_mm','dry_spell_days','heavy_rain_days','mean_temp_c','hot_days',
            'rain_intensity','dry_days','wettest_day_mm','p95_wet_day_mm',
            'longest_hot_spell','warmest_day_c','coldest_night_c',
            'hot_and_dry_days','dry_then_heavy_rain'
        ]
    },
    'MONTHS_UK': dict(enumerate('X Січень Лютий Березень Квітень Травень Червень Липень Серпень Вересень Жовтень Листопад Грудень'.split())) | {},
    'SEASON_MONTHS': {'DJF':(12,1,2), 'MAM':(3,4,5), 'JJA':(6,7,8), 'SON':(9,10,11)},
    'SEASONS_UK': {'DJF':'Зима','MAM':'Весна','JJA':'Літо','SON':'Осінь'},
})
exec(compile(ast.Module(body=functions,type_ignores=[]),'<orion_core>', 'exec'),env)

with tempfile.TemporaryDirectory() as temporary:
    env['CACHE_DIR'] = Path(temporary) / 'nasa_cache'
    start,end = 2001,2025
    dates = pd.date_range(f'{start}-01-01',f'{end}-12-31',freq='D')
    assert len(dates) == 9131
    rng = np.random.default_rng(42)
    temp = 11.0 + 9.0*np.sin(2*np.pi*dates.dayofyear.to_numpy()/365.25) + 0.05*(dates.year.to_numpy()-2001)
    rain = np.maximum(0.0, rng.exponential(4.0,len(dates)) - 2.0)
    vals = [rain, temp, temp+7, temp-6]
    payload = {
        'properties': {'parameter': {
            key: {dt.strftime('%Y%m%d'): round(float(val),3) for dt,val in zip(dates,arr)}
            for key,arr in zip(env['PARAMETERS'],vals)
        }}
    }
    lat,lon = 49.35,23.505
    url = env['NASA_URL']+'?'+urlencode(env['nasa_request_params'](lat,lon,start,end))
    env['save_cache_record'](lat,lon,start,end,payload,url)
    record = env['read_cache_record'](lat,lon,start,end)
    assert record['query']['community']=='AG'
    assert record['query']['time-standard']=='UTC'
    daily = env['fetch_nasa_power'](lat,lon,start,end)
    assert len(daily)==9131
    assert not daily.isna().any().any()
    yearly = env['aggregate_annual'](daily)
    monthly=env['aggregate_monthly'](daily)
    seasonal=env['aggregate_seasonal'](daily)
    assert len(yearly)==25 and len(monthly)==300 and len(seasonal)==99
    assert abs(yearly['rain_mm'].iloc[0] - daily[daily.index.year==2001]['rain_mm'].sum()) < .001
    assert 0 <= yearly['dry_spell_days'].max() <= 366
    assert 0 <= yearly['hot_and_dry_days'].max() <= 366
    trend=env['calc_trend'](yearly,'mean_temp_c')
    assert trend['n'] == 25 and np.isfinite(trend['slope_decade'])
    site=env['analyze_site']('Перевірка',lat,lon,start,end)
    comparison = env['evidence_comparison'](site)
    assert len(comparison)==7 and 'Зміна' in comparison.columns
    assert len(site['trends'])==14
    assert all('p_fdr' in tr and 'significant_fdr' in tr for tr in site['trends'].values())
    adjusted=env['fdr_bh']([.001,.02,.07,np.nan])
    assert np.allclose(adjusted[:3], [.003,.03,.07]) and np.isnan(adjusted[3])
    jan=env['segmented_trends'](monthly,seasonal,'rain_mm','month')
    assert len(jan)==12 and 'p_FDR' in jan
    assert env['longest_run_of_dry_days'](pd.Series([.5,.1,np.nan,.5,.5,.2,4])) == 3
    events=env['contiguous_events'](daily.iloc[:100],'rain_mm',1,min_days=3,less=True)
    assert {'Початок','Кінець','Тривалість, днів'}.issubset(events.columns)
    report=env['render_offline_showcase']([site])
    assert isinstance(report,str) and '<!doctype html>' in report.lower()
    assert 'plotly.js' in report.lower() and 'NASA POWER' in report
    assert 'Перевірка' in report
    assert not Path(temporary).joinpath('ORION_SHOWCASE.html').exists()
    cache_path=env['nasa_cache_path'](lat,lon,start,end)
    broken=json.loads(cache_path.read_text())
    broken['payload']['properties']['parameter']['T2M']['20010101'] = 999.0
    cache_path.write_text(json.dumps(broken))
    try:
        env['read_cache_record'](lat,lon,start,end)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Tampered cache was not rejected')
    print('PASS — NASA JSON parser, UTC, daily/annual/monthly/seasonal aggregation, FDR, anomalies, HTML, provenance and tamper detection')
