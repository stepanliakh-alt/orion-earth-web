"""Offline unit/smoke tests for ORION Earth 14.0; no invented data are shown in the app.
Run: python test_orion14.py
"""
import ast
import io
import math
import json
import tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, theilslopes, spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from xml.sax.saxutils import escape as xml_escape

class FakeSt:
    @staticmethod
    def cache_data(*args,**kwargs):
        def decorator(func): return func
        return decorator

src=Path(__file__).with_name('app.py').read_text(encoding='utf-8')
tree=ast.parse(src)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
constants=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('ECO_TYPES','ECO_ACTIONS','PARAMETERS','METRICS','NASA_URL') for t in n.targets)]
ns={'pd':pd,'np':np,'math':math,'io':io,'json':json,'requests':requests,'st':FakeSt,
    'datetime':datetime,'Path':Path,'kendalltau':kendalltau,'theilslopes':theilslopes,'spearmanr':spearmanr,
    'IsolationForest':IsolationForest,'StandardScaler':StandardScaler,'colors':colors,
    'TA_LEFT':TA_LEFT,'getSampleStyleSheet':getSampleStyleSheet,'ParagraphStyle':ParagraphStyle,
    'mm':mm,'pdfmetrics':pdfmetrics,'TTFont':TTFont,'SimpleDocTemplate':SimpleDocTemplate,
    'Paragraph':Paragraph,'Spacer':Spacer,'Table':Table,'TableStyle':TableStyle,
    'xml_escape':xml_escape}
exec(compile(ast.Module(body=constants+functions,type_ignores=[]),'app.py','exec'),ns)

def test_trends():
    years=np.arange(2001,2026)
    annual=pd.DataFrame({'year':years,'mean_temp_c':8+.09*(years-2001)})
    tr=ns['calc_trend'](annual,'mean_temp_c')
    assert abs(tr['slope_decade']-.9)<1e-8
    assert tr['p']<.05
    adj=ns['fdr_bh']([.01,.04,.2,np.nan])
    assert abs(adj[0]-.03)<1e-9 and np.isnan(adj[3])
    assert ns['longest_run_of_dry_days'](pd.Series([0,0,np.nan,0,0,0,5]))==3

def test_geo():
    ns['fetch_json']=lambda url,params,timeout=35: ({'elevation':[100+i*3 for i in range(9)]},'https://api.open-meteo.com/example')
    data,url=ns['fetch_elevation_profile'](49.35,23.505)
    assert len(data)==9 and data.iloc[4]['Висота, м']==112
    assert ns['osm_ecosystem_tag']({'natural':'wetland'})=='wetland'
    assert ns['osm_ecosystem_tag']({'landuse':'forest'})=='forest'
    assert ns['osm_ecosystem_tag']({'landuse':'farmland'})=='cropland'
    assert ns['osm_ecosystem_tag']({'natural':'unknown'}) is None

def test_modis():
    entries=[
        {'band':'250m_16_days_NDVI','modis_date':'A2024161','data':[7800]},
        {'band':'250m_16_days_pixel_reliability','modis_date':'A2024161','data':[0]},
        {'band':'250m_16_days_NDVI','modis_date':'A2024177','data':[6500]},
        {'band':'250m_16_days_pixel_reliability','modis_date':'A2024177','data':[3]},
        {'band':'250m_16_days_NDVI','modis_date':'A2024193','data':[-3000]},
    ]
    result=ns['parse_modis_subset']({'subset':entries})
    assert len(result)==1 and abs(result.iloc[0]['NDVI']-.78)<1e-9
    assert result.iloc[0]['Дата']==pd.Timestamp('2024-06-09')

def test_gbif():
    ns['fetch_json']=lambda url,params,timeout=35: ({'count':73,'results':[{'species':'Quercus robur','year':2023,'key':12,'decimalLatitude':49.35,'decimalLongitude':23.5}]},'https://api.gbif.org/example')
    df,count,url=ns['fetch_gbif_occurrences'](49.35,23.505)
    assert count==73 and df.iloc[0]['Таксон']=='Quercus robur'

def test_evidence_and_pdf():
    tr={key:{'slope_decade':0.2,'p':.001,'p_fdr':.01,'significant_fdr':True} for key in ns['METRICS']}
    site={'name':'Дрогобич, Україна','lat':49.35,'lon':23.505,
          'annual':pd.DataFrame({'year':[2023,2024,2025]}),'trends':tr,
          'daily':pd.DataFrame({'rain_mm':np.ones(365*3),'tmax_c':np.ones(365*3)*25},index=pd.date_range('2023-01-01','2025-12-30'))}
    ndvi=pd.DataFrame({'Дата':pd.to_datetime(['2024-06-01','2024-06-17','2024-07-03','2025-06-01','2025-06-17','2025-07-03']),
                       'Рік':[2024]*3+[2025]*3,'NDVI':[.7,.71,.72,.58,.60,.62],
                       'Pixel reliability':[0]*6})
    climate_ndvi=ns['summer_ndvi_climate_table'](site['daily'],ndvi)
    assert len(climate_ndvi)==2 and 'Опади (літо), мм' in climate_ndvi.columns
    assert ns['ndvi_climate_associations'](climate_ndvi).empty
    evidence=ns['ecosystem_evidence'](site,'forest',ndvi)
    assert len(evidence)==5 and any(x['Сигнал']=='Зміна NDVI (два літа)' for x in evidence)
    pdf=ns['build_ecosystem_pdf'](site,'forest',evidence,ndvi=ndvi,gbif_count=73)
    assert pdf.startswith(b'%PDF-') and len(pdf)>2000
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'smoke_ecosystem_report.pdf'
        path.write_bytes(pdf)
        assert path.stat().st_size==len(pdf)

if __name__=='__main__':
    test_trends(); print('✓ Annual trends, FDR, dry spell')
    test_geo(); print('✓ Copernicus elevation, OSM categories')
    test_modis(); print('✓ MODIS NDVI QA and scaling')
    test_gbif(); print('✓ GBIF response parsing')
    test_evidence_and_pdf(); print('✓ Ecosystem recommendations and Cyrillic PDF')
    print('ALL TESTS PASSED (mocked external services; live APIs require internet)')
