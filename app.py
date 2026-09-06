import io, json, math, os, threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np, pandas as pd, requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

JST=timezone(timedelta(hours=9)); app=FastAPI(title="Rain Rarity Flood Depth Live",version="0.1.0"); app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
STATIC_DIR=os.path.join(os.path.dirname(__file__),"static"); SESSION=requests.Session(); SESSION.headers.update({"User-Agent":"RainRarityFloodDepth/0.1 research-prototype"})
MODEL_FEATURES=["rain_1h_mm","rain_3h_mm","rain_24h_mm","rain_upper_3h_mm","rarity_e90m_y","rarity_r3h_y","rarity_r6h_y","rarity_r24h_y","elevation_m","relative_elevation_m"]
MODEL_LOCK=threading.Lock(); MODELS={"p50":None,"p95":None,"meta":None}
NIED_SERVICES={"rarity_e90m_y":"e90mrp","rarity_e72h_y":"e72hrp","rarity_r3h_y":"r03hrp","rarity_r6h_y":"r06hrp","rarity_r24h_y":"r24hrp"}
NIED_BASE="https://midoplat2.bosai.go.jp/webgis/rest/services/rainrp/{service}/ImageServer"; JMA_AMEDAS_TABLE="https://www.jma.go.jp/bosai/amedas/const/amedastable.json"; JMA_POINT="https://www.jma.go.jp/bosai/amedas/data/point/{station}/{date}_00.json"; GSI_ELEV="https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"; OPEN_METEO="https://api.open-meteo.com/v1/forecast"; _amedas_cache=None
class SnapshotRequest(BaseModel):
 lat:float=Field(ge=-90,le=90); lon:float=Field(ge=-180,le=180); hours_ahead:int=Field(default=3,ge=1,le=6)
def safe_get(url,params=None,timeout=12):
 r=SESSION.get(url,params=params,timeout=timeout); r.raise_for_status(); return r
def get_amedas_table():
 global _amedas_cache
 if _amedas_cache is None:_amedas_cache=safe_get(JMA_AMEDAS_TABLE).json()
 return _amedas_cache
def haversine(a,b,c,d):
 r=6371.;p1,p2=math.radians(a),math.radians(c);dp,dl=math.radians(c-a),math.radians(d-b);x=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2;return 2*r*math.asin(math.sqrt(x))
def nearest_amedas(lat,lon):
 best=None
 for sid,m in get_amedas_table().items():
  try:
   a=float(m['lat'][0])+float(m['lat'][1])/60;b=float(m['lon'][0])+float(m['lon'][1])/60;d=haversine(lat,lon,a,b)
   if best is None or d<best[2]:best=(sid,m,d)
  except:pass
 if best is None:raise RuntimeError('AMeDAS station not found')
 return best
def fetch_jma_recent(lat,lon):
 sid,m,dist=nearest_amedas(lat,lon);now=datetime.now(JST);merged={}
 for d in reversed([(now-timedelta(days=i)).strftime('%Y%m%d') for i in range(2)]):
  try:merged.update(safe_get(JMA_POINT.format(station=sid,date=d)).json())
  except:pass
 if not merged:raise RuntimeError('JMA AMeDAS point data unavailable')
 rows=[]
 for ts,rec in merged.items():
  try:dt=datetime.strptime(ts,'%Y%m%d%H%M%S').replace(tzinfo=JST)
  except:continue
  p10=rec.get('precipitation10m');p1=rec.get('precipitation1h');v10=float(p10[0]) if isinstance(p10,list) and p10 and p10[0] is not None else 0.;v1=float(p1[0]) if isinstance(p1,list) and p1 and p1[0] is not None else None;rows.append((dt,max(0.,v10),v1))
 rows.sort();cut=now-timedelta(hours=24,minutes=20);rows=[r for r in rows if cut<=r[0]<=now+timedelta(minutes=10)]
 if not rows:raise RuntimeError('No recent AMeDAS rainfall rows')
 s=pd.DataFrame(rows,columns=['time','p10','p1h']);one=s.tail(6).p10.sum();three=s.tail(18).p10.sum();day=s.tail(144).p10.sum();latest=s.p1h.dropna().iloc[-1] if s.p1h.notna().any() else one
 return {'source':'JMA AMeDAS','station_id':sid,'station_name':m.get('kjName') or m.get('enName') or sid,'distance_km':round(dist,2),'rain_1h_mm':round(float(max(one,latest or 0)),2),'rain_3h_mm':round(float(three),2),'rain_24h_mm':round(float(day),2),'series_10m':[{'t':t.isoformat(),'mm':float(v)} for t,v in zip(s.time,s.p10)]}
def fetch_openmeteo_recent(lat,lon):
 j=safe_get(OPEN_METEO,{'latitude':lat,'longitude':lon,'hourly':'precipitation','past_days':1,'forecast_days':1,'timezone':'Asia/Tokyo'}).json();now=datetime.now(JST).replace(tzinfo=None);data=[(datetime.fromisoformat(t),float(v or 0)) for t,v in zip(j.get('hourly',{}).get('time',[]),j.get('hourly',{}).get('precipitation',[])) if datetime.fromisoformat(t)<=now];v=[x[1] for x in data]
 return {'source':'Open-Meteo fallback (not gauge observation)','station_id':None,'station_name':None,'distance_km':None,'rain_1h_mm':round(sum(v[-1:]),2),'rain_3h_mm':round(sum(v[-3:]),2),'rain_24h_mm':round(sum(v[-24:]),2),'series_10m':[{'t':dt.replace(tzinfo=JST).isoformat(),'mm':x/6} for dt,x in data[-24:] for _ in range(6)]}
def conservative_rain_upper(series,hours_ahead=3):
 vals=np.array([max(0.,float(x.get('mm',0))) for x in series]);
 if len(vals)<6:return {'rain_upper_3h_mm':0.,'recent_peak_1h_mm':0.}
 recent=vals[-min(36,len(vals)):];rolling=np.convolve(vals,np.ones(6),mode='valid');peak=float(np.max(rolling[-min(36,len(rolling)):])) ;cur=float(vals[-6:].sum());x=np.arange(len(recent),dtype=float);slope=float(np.polyfit(x,recent,1)[0]) if len(recent)>=6 else 0.;trend=max(0.,slope*36);burst=float(np.quantile(recent,.95)*6);total=float(np.clip((max(cur,peak,burst)+trend)*hours_ahead,0,500));return {'rain_upper_3h_mm':round(total,2),'recent_peak_1h_mm':round(peak,2)}
def nied_latest_raster_id(service):
 try:
  j=safe_get(NIED_BASE.format(service=service)+'/query',{'where':'latest=1','outFields':'objectid,data_datetime,latest','returnGeometry':'false','orderByFields':'data_datetime DESC','resultRecordCount':1,'f':'json'}).json();f=j.get('features',[]);return int(f[0]['attributes']['objectid']) if f else None
 except:return None
def fetch_nied_rarity(lat,lon,service):
 base=NIED_BASE.format(service=service);rid=nied_latest_raster_id(service);p={'geometry':json.dumps({'x':lon,'y':lat,'spatialReference':{'wkid':4326}},separators=(',',':')),'geometryType':'esriGeometryPoint','returnGeometry':'false','returnCatalogItems':'false','renderingRule':json.dumps({'rasterFunction':'rainrp'}),'f':'json'}
 if rid is not None:p['mosaicRule']=json.dumps({'mosaicMethod':'esriMosaicLockRaster','lockRasterIds':[rid]})
 try:
  v=safe_get(base+'/identify',p).json().get('value');return None if v in (None,'NoData','NODATA','') else round(float(v),2)
 except:return None
def get_elevation(lat,lon):
 try:
  e=safe_get(GSI_ELEV,{'lon':lon,'lat':lat,'outtype':'JSON'}).json().get('elevation');return None if e in (None,'-----') else float(e)
 except:return None
def elevation_context(lat,lon):
 c=get_elevation(lat,lon)
 if c is None:return {'elevation_m':None,'relative_elevation_m':None}
 da=.00225;do=.00225/max(.2,math.cos(math.radians(lat)));n=[get_elevation(lat+a*da,lon+b*do) for a,b in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]];n=[x for x in n if x is not None];rel=c-float(np.median(n)) if n else None;return {'elevation_m':round(c,2),'relative_elevation_m':round(rel,2) if rel is not None else None}
def hazard_screen(rain,rarity,terrain):
 rp=max([v for v in rarity.values() if isinstance(v,(int,float))] or [0]);upper=rain.get('rain_upper_3h_mm',0) or 0;rel=terrain.get('relative_elevation_m');score=min(100,20*math.log10(max(1,rp))+min(50,upper/3)+(15 if rel is not None and rel<-1 else 0));level='低い' if score<25 else '中' if score<50 else '高い' if score<75 else '極めて高い';return {'score':round(score,1),'level':level}
def predict_depth(features):
 with MODEL_LOCK:m50,m95,meta=MODELS['p50'],MODELS['p95'],MODELS['meta']
 if m50 is None:return {'available':False,'reason':'実浸水深CSVでモデルを学習してください'}
 x=np.array([[float(features.get(c) if features.get(c) is not None else meta['medians'].get(c,0)) for c in MODEL_FEATURES]]);p50=max(0,float(m50.predict(x)[0]));p95=max(p50,float(m95.predict(x)[0]));return {'available':True,'p50_m':round(p50,3),'p95_m':round(p95,3)}
@app.get('/')
def root():return FileResponse(os.path.join(STATIC_DIR,'index.html'))
@app.get('/health')
def health():return {'ok':True,'model_trained':MODELS['p50'] is not None}
@app.post('/api/snapshot')
def snapshot(q:SnapshotRequest):
 try:rain=fetch_jma_recent(q.lat,q.lon)
 except:
  try:rain=fetch_openmeteo_recent(q.lat,q.lon)
  except Exception as e:raise HTTPException(502,f'Rainfall unavailable: {e}')
 rain.update(conservative_rain_upper(rain.get('series_10m',[]),q.hours_ahead));rarity={k:fetch_nied_rarity(q.lat,q.lon,s) for k,s in NIED_SERVICES.items()};terrain=elevation_context(q.lat,q.lon);features={**rain,**rarity,**terrain};return {'timestamp':datetime.now(JST).isoformat(),'lat':q.lat,'lon':q.lon,'rainfall':rain,'rarity':rarity,'terrain':terrain,'hazard_screen':hazard_screen(rain,rarity,terrain),'depth':predict_depth(features)}
@app.post('/api/train')
async def train(file:UploadFile=File(...)):
 try:df=pd.read_csv(io.BytesIO(await file.read()))
 except Exception as e:raise HTTPException(400,f'CSV read error: {e}')
 req=MODEL_FEATURES+['depth_m'];miss=[c for c in req if c not in df.columns]
 if miss:raise HTTPException(400,detail={'missing_columns':miss,'required':req})
 w=df[req].apply(pd.to_numeric,errors='coerce');med=w[MODEL_FEATURES].median(numeric_only=True).fillna(0).to_dict()
 for c in MODEL_FEATURES:w[c]=w[c].fillna(med[c])
 w=w.dropna(subset=['depth_m'])
 if len(w)<30:raise HTTPException(400,'最低30行の実浸水深データを推奨します')
 X=w[MODEL_FEATURES].to_numpy(float);y=w.depth_m.to_numpy(float);rng=np.random.default_rng(42);idx=np.arange(len(w));rng.shuffle(idx);split=max(20,int(len(idx)*.8));tr,te=idx[:split],idx[split:]
 if len(te)<5:te=tr[-5:];tr=tr[:-5]
 m50=GradientBoostingRegressor(loss='quantile',alpha=.5,n_estimators=250,max_depth=3,learning_rate=.04,random_state=42);m95=GradientBoostingRegressor(loss='quantile',alpha=.95,n_estimators=300,max_depth=3,learning_rate=.04,random_state=43);m50.fit(X[tr],y[tr]);m95.fit(X[tr],y[tr]);p50=np.maximum(0,m50.predict(X[te]));p95=np.maximum(p50,m95.predict(X[te]));meta={'n':int(len(w)),'features':MODEL_FEATURES,'target':'depth_m','mae_p50_holdout':round(float(mean_absolute_error(y[te],p50)),4),'p95_coverage_holdout':round(float(np.mean(y[te]<=p95)),4),'medians':{k:float(v) for k,v in med.items()},'trained_at':datetime.now(JST).isoformat()}
 with MODEL_LOCK:MODELS.update({'p50':m50,'p95':m95,'meta':meta})
 return {'ok':True,'model':meta,'note':'P95 coverageは独立検証データで再評価してください。'}
@app.get('/api/template')
def template():
 cols=MODEL_FEATURES+['depth_m'];return JSONResponse({'columns':cols,'csv_header':','.join(cols)})
if __name__=='__main__':
 import uvicorn;uvicorn.run(app,host='0.0.0.0',port=int(os.getenv('PORT','8000')))
