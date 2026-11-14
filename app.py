"""ORION Earth WEB 5.0 — all sections visible, NASA POWER evidence-based climate intelligence.
Run: streamlit run app.py
Requires internet to retrieve authentic NASA data. No simulated data is displayed.
"""
from __future__ import annotations

import io
import json
import calendar
import hashlib
import html
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import plotly.io as pio

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import kendalltau, theilslopes
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ORION Earth | NASA Climate Intelligence", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "nasa_cache"
PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN"]
LOCATIONS = {
    "Дрогобич, Україна": (49.3500, 23.5050),
    "Львів, Україна": (49.8397, 24.0297),
    "Київ, Україна": (50.4501, 30.5234),
    "Одеса, Україна": (46.4825, 30.7233),
    "Афіни, Греція": (37.9838, 23.7275),
    "Найробі, Кенія": (-1.2921, 36.8219),
    "Делі, Індія": (28.6139, 77.2090),
    "Сан-Паулу, Бразилія": (-23.5505, -46.6333),
    "Фінікс, США": (33.4484, -112.0740),
    "Сідней, Австралія": (-33.8688, 151.2093),
}
METRICS = {
    "rain_mm": ("Сума опадів", "мм/рік", "мм/10 років"),
    "dry_spell_days": ("Найдовший сухий період", "днів", "днів/10 років"),
    "heavy_rain_days": ("Дні з опадами ≥20 мм", "днів/рік", "днів/10 років"),
    "mean_temp_c": ("Середня температура", "°C", "°C/10 років"),
    "hot_days": ("Спекотні дні (Tmax ≥30°C)", "днів/рік", "днів/10 років"),
    "rain_intensity": ("Інтенсивність у вологі дні", "мм/вологий день", "мм/10 років"),
    "dry_days": ("Дні з опадами <1 мм", "днів/рік", "днів/10 років"),
    "wettest_day_mm": ("Найсильніший дощ за добу", "мм/день", "мм/10 років"),
    "p95_wet_day_mm": ("95-й перцентиль дощових днів", "мм/день", "мм/10 років"),
    "longest_hot_spell": ("Найдовша серія спекотних днів", "днів", "днів/10 років"),
    "warmest_day_c": ("Максимум Tmax за рік", "°C", "°C/10 років"),
    "coldest_night_c": ("Мінімум Tmin за рік", "°C", "°C/10 років"),
    "hot_and_dry_days": ("Спека і бездощів'я в один день", "днів/рік", "днів/10 років"),
    "dry_then_heavy_rain": ("Злива після ≥7 сухих днів", "подій/рік", "подій/10 років"),
}

# Consistent dark palette for all Plotly figures (theme is also configured by the launcher).
px.defaults.template = "plotly_dark"

st.markdown("""
<style>
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 90% 5%, #122c44 0, #0a1222 40%, #080e1b 100%); color: #e7f0fb; }
    [data-testid="stSidebar"] { background: #0d1a2b; border-right: 1px solid #22354b; }
    .block-container { padding-top: 1.15rem; max-width: 1480px; }
    h1,h2,h3 { letter-spacing: -.025em; color: #e9f5ff !important; }
    h1 { font-size: clamp(1.8rem, 3vw, 3rem) !important; }
    div[data-testid="stMetric"] { background: linear-gradient(145deg,#12263b,#0d1a2d);
       border: 1px solid #2a4359; padding: 16px; border-radius: 14px; }
    div[data-testid="stMetricValue"] { color: #70d7f4; }
    div[data-testid="stMetricLabel"] { color: #b9cce2; }
    html { scroll-behavior: smooth; }
    .orion-nav { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 9px; margin: 12px 0 24px; }
    .orion-nav a { display: block; border: 1px solid #27485f; border-radius: 12px;
      padding: 11px 12px; background: #12253b; color: #b5f4ff !important;
      font-size: .88rem; font-weight: 600; text-decoration: none !important; }
    .orion-nav a:hover { border-color: #4ad7dc; background: #17324a; }
    .orion-side-nav a { display: block; padding: 7px 12px; margin: 3px 0;
      border-left: 3px solid #2bd5d1; border-radius: 6px; color: #d3eaf5 !important;
      background: #15293d; text-decoration:none !important; }
    .orion-side-nav a:hover { color:#55ecf0 !important; background:#1d3b55; }
    .orion-section-head { border-top: 1px solid #29435e; margin: 34px 0 14px;
      padding-top: 25px; scroll-margin-top: 50px; }
    .orion-section-head span {color: #49dbe2; letter-spacing:.13em; font-size:.75rem; font-weight:800;}
    .orion-section-head h2 {font-size:1.65rem; margin:7px 0;}
    @media(max-width:650px) { .orion-nav {grid-template-columns: 1fr;} }
    .caption-box { padding: 12px 16px; border-left: 3px solid #4ccbeb;
       background: rgba(56,189,248,.07); border-radius: 6px; }
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def longest_run_of_dry_days(rain: pd.Series) -> int:
    """Longest consecutive run of <1 mm/day. Missing observations break the run."""
    longest = current = 0
    for v in rain:
        if pd.isna(v):
            current = 0
        elif v < 1.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def nasa_request_params(lat: float, lon: float, start_year: int, end_year: int) -> dict:
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "AG", "longitude": round(float(lon), 5),
        "latitude": round(float(lat), 5),
        "start": f"{int(start_year)}0101", "end": f"{int(end_year)}1231",
        "format": "JSON", "time-standard": "UTC",
    }


def nasa_cache_path(lat: float, lon: float, start_year: int, end_year: int) -> Path:
    params = nasa_request_params(lat, lon, start_year, end_year)
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"nasa_power_{key}.json"


def validated_parameters(payload: dict):
    if not isinstance(payload, dict):
        raise RuntimeError("Некоректний JSON NASA POWER")
    parameters = payload.get("properties", {}).get("parameter", {})
    if not isinstance(parameters, dict) or not all(isinstance(parameters.get(k), dict) and parameters.get(k) for k in PARAMETERS):
        raise RuntimeError(f"Відповідь NASA POWER не містить усіх полів: {PARAMETERS}")
    return parameters


def read_cache_record(lat: float, lon: float, start_year: int, end_year: int):
    path = nasa_cache_path(lat, lon, start_year, end_year)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        params = nasa_request_params(lat, lon, start_year, end_year)
        assert record["source"] == "NASA POWER Daily Point API"
        assert record["query"] == params
        payload = record["payload"]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        assert record["payload_sha256"] == digest
        validated_parameters(payload)
        return record
    except (KeyError, ValueError, TypeError, AssertionError, RuntimeError) as exc:
        raise RuntimeError(f"Пошкоджений або недостовірний кеш NASA ({path.name}). Видали файл та перезавантаж: {exc}") from exc


def save_cache_record(lat: float, lon: float, start_year: int, end_year: int, payload: dict, api_url: str):
    validated_parameters(payload)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = nasa_request_params(lat, lon, start_year, end_year)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    record = {
        "source": "NASA POWER Daily Point API",
        "api_url": api_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "query": params, "payload_sha256": digest, "payload": payload,
    }
    path = nasa_cache_path(lat, lon, start_year, end_year)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(record, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp_path.replace(path)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_nasa_power(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """Daily NASA POWER data only. Network first time; persistent verified local copy later."""
    record = read_cache_record(lat, lon, start_year, end_year)
    if record is not None:
        payload = record["payload"]
    else:
        params = nasa_request_params(lat, lon, start_year, end_year)
        try:
            with requests.Session() as session:
                response = session.get(NASA_URL, params=params, timeout=120)
                response.raise_for_status()
                payload = response.json()
                validated_parameters(payload)
                save_cache_record(lat, lon, start_year, end_year, payload, response.url)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                "Не вдалося отримати фактичні дані NASA POWER. "
                "Перевір інтернет, доступність POWER або вибери раніше завантажений набір. "
                f"Причина: {exc}"
            ) from exc
    source = validated_parameters(payload)
    df = pd.DataFrame(source)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()].sort_index()
    full_dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    df = df.reindex(full_dates)
    df.index.name = "date"
    for col in PARAMETERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace({-999: np.nan, -999.0: np.nan})
    df = df.rename(columns={"PRECTOTCORR": "rain_mm", "T2M": "temp_c", "T2M_MAX": "tmax_c", "T2M_MIN": "tmin_c"})
    df["rain_mm"] = df["rain_mm"].where(df["rain_mm"] >= 0)
    if df.empty or not df["rain_mm"].notna().any():
        raise RuntimeError("NASA не повернула достатньо коректних даних для цієї точки.")
    return df[["rain_mm", "temp_c", "tmax_c", "tmin_c"]]


def count_dry_to_rain_events(rain: pd.Series, dry_threshold: float = 1, min_dry_days: int = 7, heavy_threshold: float = 20) -> int:
    """Number of heavy-rain days following >=7 dry days without a missing day."""
    n, streak = 0, 0
    for value in rain:
        if pd.isna(value):
            streak = 0
        elif float(value) < dry_threshold:
            streak += 1
        else:
            if float(value) >= heavy_threshold and streak >= min_dry_days:
                n += 1
            streak = 0
    return n


def aggregate_annual(daily: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
    """Annual indices; exclude years if any required series has <90% day coverage."""
    result = []
    for year, g in daily.groupby(daily.index.year):
        expected = 366 if calendar.isleap(int(year)) else 365
        coverage = min(g[c].count() / expected for c in ["rain_mm", "temp_c", "tmax_c", "tmin_c"])
        if coverage < min_coverage:
            continue
        wet = g.loc[g["rain_mm"] >= 1, "rain_mm"]
        result.append({
            "year": int(year),
            "coverage": float(coverage),
            "rain_mm": float(g["rain_mm"].sum()),
            "dry_spell_days": int(longest_run_of_dry_days(g["rain_mm"])),
            "heavy_rain_days": int((g["rain_mm"] >= 20).sum()),
            "mean_temp_c": float(g["temp_c"].mean()),
            "hot_days": int((g["tmax_c"] >= 30).sum()),
            "rain_intensity": float(wet.mean()) if len(wet) else np.nan,
            "dry_days": int((g["rain_mm"] < 1).sum()),
            "wettest_day_mm": float(g["rain_mm"].max()),
            "p95_wet_day_mm": float(wet.quantile(.95)) if len(wet) >= 5 else np.nan,
            "longest_hot_spell": int(longest_boolean_run(g["tmax_c"] >= 30)),
            "warmest_day_c": float(g["tmax_c"].max()),
            "coldest_night_c": float(g["tmin_c"].min()),
            "hot_and_dry_days": int(((g["tmax_c"] >= 30) & (g["rain_mm"] < 1)).sum()),
            "dry_then_heavy_rain": int(count_dry_to_rain_events(g["rain_mm"])),
        })
    return pd.DataFrame(result).sort_values("year").reset_index(drop=True) if result else pd.DataFrame()


def calc_trend(annual: pd.DataFrame, field: str) -> dict:
    """Mann–Kendall rank trend + Sen slope (scipy Theil–Sen) and 95% CI."""
    d = annual[["year", field]].dropna()
    if len(d) < 10:
        return {"n": len(d), "slope_decade": np.nan, "p": np.nan,
                "lower_decade": np.nan, "upper_decade": np.nan, "significant": False,
                "tau": np.nan}
    x, y = d["year"].to_numpy(), d[field].to_numpy()
    if np.unique(y).size == 1:
        return {"n": len(d), "slope_decade": 0.0, "p": 1.0,
                "lower_decade": 0.0, "upper_decade": 0.0, "significant": False,
                "tau": 0.0}
    tau, p = kendalltau(x, y)
    slope, intercept, lower, upper = theilslopes(y, x, alpha=0.95)
    return {"n": len(d), "slope_decade": float(slope * 10),
            "p": float(p), "lower_decade": float(lower * 10),
            "upper_decade": float(upper * 10),
            "significant": bool(np.isfinite(p) and p < 0.05),
            "tau": float(tau)}


def label_trend(result: dict, unit: str) -> str:
    slope, p = result["slope_decade"], result["p"]
    if not np.isfinite(slope):
        return "Недостатньо років для тесту"
    direction = "зростання" if slope > 0 else "зниження" if slope < 0 else "без зміни"
    corrected = result.get("p_fdr", np.nan)
    certainty = "значущий після FDR" if result.get("significant_fdr", False) else "не підтверджено після FDR"
    return f"{direction} {slope:+.2f} {unit}; p={p:.3g}; p_FDR={corrected:.3g} — {certainty}"


def detect_anomalies(annual: pd.DataFrame) -> pd.DataFrame:
    """Unsupervised exploratory anomalies, NOT risk forecasts."""
    out = annual.copy()
    cols = ["rain_mm", "dry_spell_days", "heavy_rain_days", "mean_temp_c", "hot_days"]
    full = out[cols].dropna()
    out["anomaly"] = False
    out["anomaly_score"] = np.nan
    if len(full) < 12:
        return out
    scaled = StandardScaler().fit_transform(full)
    model = IsolationForest(n_estimators=300, contamination=0.15, random_state=42)
    model.fit(scaled)
    out.loc[full.index, "anomaly"] = model.predict(scaled) == -1
    out.loc[full.index, "anomaly_score"] = model.decision_function(scaled)
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def detect_daily_anomalies(daily: pd.DataFrame) -> pd.DataFrame:
    """Find unusual *days*, not years. Normalize by calendar month to reduce season bias.

    IsolationForest is unsupervised exploratory outlier detection; flagged dates
    are not natural-disaster forecasts or probabilities.
    """
    sub = daily[['rain_mm', 'tmax_c', 'tmin_c']].dropna().copy()
    if len(sub) < 365:
        return pd.DataFrame(columns=['Дата', 'Опади, мм', 'Tmax, °C', 'Tmin, °C', 'Оцінка аномальності'])
    features = pd.DataFrame(index=sub.index)
    features['log_rain'] = np.log1p(sub['rain_mm'])
    features['tmax'] = sub['tmax_c']
    features['tmin'] = sub['tmin_c']
    # Months are used for climatological reference, not to replace daily records.
    for col in features.columns:
        v = features[col]
        months = v.index.month
        mu = v.groupby(months).transform('mean')
        sigma = v.groupby(months).transform('std').replace(0, np.nan).fillna(1)
        features[col] = (v - mu) / sigma
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(features) < 365:
        return pd.DataFrame(columns=['Дата', 'Опади, мм', 'Tmax, °C', 'Tmin, °C', 'Оцінка аномальності'])
    model = IsolationForest(n_estimators=160, contamination=.02, random_state=42, n_jobs=-1)
    labels = model.fit_predict(features)
    scores = model.decision_function(features)
    # Supplement multivariate anomalies with objectively rare single-day
    # extremes: the combination model alone may miss an isolated heavy storm.
    aligned = sub.loc[features.index]
    month = aligned.index.month
    rain_cut = aligned['rain_mm'].groupby(month).transform(lambda x: x.quantile(.995))
    hot_cut = aligned['tmax_c'].groupby(month).transform(lambda x: x.quantile(.995))
    cold_cut = aligned['tmin_c'].groupby(month).transform(lambda x: x.quantile(.005))
    heavy = aligned['rain_mm'] > rain_cut
    hot = aligned['tmax_c'] > hot_cut
    cold = aligned['tmin_c'] < cold_cut
    include = (labels == -1) | heavy.to_numpy() | hot.to_numpy() | cold.to_numpy()
    unusual = aligned.iloc[np.flatnonzero(include)].copy()
    unusual['Оцінка аномальності'] = scores[include]
    reasons = []
    for dt in unusual.index:
        why = []
        if heavy.loc[dt]: why.append('Екстремальні опади')
        if hot.loc[dt]: why.append('Екстремальна спека')
        if cold.loc[dt]: why.append('Екстремальний холод')
        if not why: why.append('Нетипове поєднання добових показників')
        reasons.append('; '.join(why))
    unusual['Причина позначки'] = reasons
    unusual.index.name = 'Дата'
    unusual = unusual.rename(columns={'rain_mm':'Опади, мм', 'tmax_c':'Tmax, °C', 'tmin_c':'Tmin, °C'})
    return unusual.reset_index().sort_values('Оцінка аномальності').reset_index(drop=True)

def analyze_site(name: str, lat: float, lon: float, start_year: int, end_year: int) -> dict:
    daily = fetch_nasa_power(lat, lon, start_year, end_year)
    annual = aggregate_annual(daily)
    if len(annual) < 10:
        raise RuntimeError(f"{name}: тільки {len(annual)} повних років; потрібно хоча б 10.")
    trends = {key: calc_trend(annual, key) for key in METRICS}
    # Correct the whole annual metric family for multiple testing (exploratory FDR).
    adjusted_p = fdr_bh([trends[key]["p"] for key in METRICS])
    for key, p_adjusted in zip(METRICS, adjusted_p):
        trends[key]["p_fdr"] = float(p_adjusted) if np.isfinite(p_adjusted) else np.nan
        trends[key]["significant_fdr"] = bool(np.isfinite(p_adjusted) and p_adjusted < 0.05)
    annual = detect_anomalies(annual)
    rain, dry = trends["rain_mm"], trends["dry_spell_days"]
    paradox_candidate = bool(rain["slope_decade"] > 0 and dry["slope_decade"] > 0)
    paradox_confirmed = bool(paradox_candidate and rain["significant_fdr"] and dry["significant_fdr"])
    return dict(name=name, lat=lat, lon=lon, daily=daily, annual=annual,
                trends=trends, paradox_candidate=paradox_candidate,
                paradox_confirmed=paradox_confirmed)


def evidence_comparison(site: dict) -> pd.DataFrame:
    """Observed change between first/last five COMPLETE years (descriptive, not causal)."""
    annual = site['annual'].sort_values('year')
    if len(annual) < 10:
        return pd.DataFrame()
    first, last = annual.head(5), annual.tail(5)
    fields = ['mean_temp_c', 'hot_days', 'hot_and_dry_days',
              'rain_mm', 'dry_spell_days', 'heavy_rain_days',
              'dry_then_heavy_rain']
    rows = []
    for field in fields:
        first_value, last_value = float(first[field].mean()), float(last[field].mean())
        rows.append({
            'Показник': METRICS[field][0],
            f"{int(first['year'].min())}–{int(first['year'].max())}": round(first_value, 2),
            f"{int(last['year'].min())}–{int(last['year'].max())}": round(last_value, 2),
            'Зміна': round(last_value - first_value, 2),
            'Значущий довгостроковий тренд (FDR)':
                bool(site['trends'][field]['significant_fdr']),
        })
    return pd.DataFrame(rows)


def implications(site: dict) -> list[tuple[str, str]]:
    """Applications of robust trends, never causal or operational warnings."""
    t = site["trends"]
    notes = []
    if site["paradox_confirmed"]:
        notes.append(("🚜 Сільське господарство", "Опади та максимальна тривалість сухих періодів одночасно зростають. Доцільно перевірити сезонний розподіл опадів, вологість ґрунту та потребу у зрошенні."))
    elif site["paradox_candidate"]:
        notes.append(("🧪 Гіпотеза для дослідження", "Виявлено напрямок «більше опадів — довші сухі періоди», але статистичної значущості обох трендів немає. Не робіть категоричних висновків."))
    if t["heavy_rain_days"]["significant_fdr"] and t["heavy_rain_days"]["slope_decade"] > 0:
        notes.append(("🏙️ Міське планування", "Частішають дні з ≥20 мм опадів. Це підстава перевірити пропускну здатність дренажу та локальні карти затоплення, а не прогноз конкретної повені."))
    if t["mean_temp_c"]["significant_fdr"] and t["mean_temp_c"]["slope_decade"] > 0:
        notes.append(("🌡️ Температурний режим", "Підтверджено зростання середньорічної температури. Перевірте розподіл добових максимумів і мінімумів, спеку по сезонах і локальні плани адаптації. Сам тренд не є прогнозом впливу на здоров’я."))
    if t["hot_days"]["significant_fdr"] and t["hot_days"]["slope_decade"] > 0:
        notes.append(("🏥 Громадське здоров’я", "Кількість днів з Tmax ≥30°C зростає. Це аргумент для теплових планів міст, затінення й захисту вразливих груп."))
    if t["rain_mm"]["significant_fdr"] and t["rain_mm"]["slope_decade"] < 0:
        notes.append(("💧 Водні ресурси", "Сума опадів знижується. Потрібно перевірити випаровування, запаси води та попит, перш ніж робити висновок про дефіцит."))
    if t["hot_and_dry_days"]["significant_fdr"] and t["hot_and_dry_days"]["slope_decade"] > 0:
        notes.append(("🌾 Комбіновані спекотні та сухі дні", "Частішає одночасне виникнення Tmax ≥30 °C і добових опадів <1 мм. Перевірити вологість ґрунту та вплив на культури перед оцінкою посухи."))
    if t["dry_then_heavy_rain"]["significant_fdr"] and t["dry_then_heavy_rain"]["slope_decade"] > 0:
        notes.append(("🌧️ Злива після сухого періоду", "Почастішали зливи ≥20 мм одразу після не менш ніж семи сухих днів. Це сигнал для перевірки стоку та ґрунтових умов, не прогноз підтоплення."))
    if not notes:
        notes.append(("🔎 Моніторинг", "За заданими правилами виражених сигналів не виявлено. Відсутність значущого тренду не означає відсутність кліматичних ризиків."))
    return notes


def trend_plot(site: dict, field: str):
    annual, tr = site["annual"], site["trends"][field]
    name, units, _ = METRICS[field]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annual["year"], y=annual[field], mode="lines+markers",
                             name="Річні значення", line=dict(color="#38bdf8", width=2)))
    valid = annual[["year", field]].dropna()
    if len(valid) >= 10 and np.isfinite(tr["slope_decade"]):
        slope_year = tr["slope_decade"] / 10
        mid_year = float(np.median(valid["year"]))
        intercept = float(np.median(valid[field] - slope_year * (valid["year"] - mid_year)))
        fit = intercept + slope_year * (valid["year"] - mid_year)
        fig.add_trace(go.Scatter(x=valid["year"], y=fit, name="Лінія Сена",
                                 line=dict(color="#fb923c", dash="dash", width=3)))
    fig.update_layout(title=name, xaxis_title="Рік", yaxis_title=units, height=390,
                      margin=dict(t=55, b=35, l=45, r=20), legend=dict(orientation="h", y=-.2))
    return fig


def make_comparison(sites: list[dict], field: str):
    rows = []
    for site in sites:
        t = site["trends"][field]
        rows.append(dict(Регіон=site["name"], Зміна=t["slope_decade"],
                         Статус="FDR < 0.05" if t["significant_fdr"] else "не підтверджено"))
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="Регіон", y="Зміна", color="Статус", color_discrete_map={
        "FDR < 0.05": "#14b8a6", "не підтверджено": "#64748b"},
        title=f"{METRICS[field][0]}: зміна за 10 років")
    fig.update_layout(yaxis_title=METRICS[field][2], xaxis_title="", height=400)
    return fig



# ─────────────────── DIFF / TEMPORAL METRICS ───────────────────
MONTHS_UK = {1:'Січень',2:'Лютий',3:'Березень',4:'Квітень',5:'Травень',6:'Червень',
             7:'Липень',8:'Серпень',9:'Вересень',10:'Жовтень',11:'Листопад',12:'Грудень'}
SEASONS_UK = {'DJF':'Зима (DJF)', 'MAM':'Весна (MAM)', 'JJA':'Літо (JJA)', 'SON':'Осінь (SON)'}
SEASON_MONTHS = {'DJF':(12,1,2), 'MAM':(3,4,5), 'JJA':(6,7,8), 'SON':(9,10,11)}
SEG_FIELDS = {
    'rain_mm':'Опади, мм за період',
    'dry_days':'Сухі дні (<1 мм)',
    'heavy_rain_days':'Дні зі зливою (≥20 мм)',
    'hot_days':'Спекотні дні (Tmax ≥30°C)',
    'wettest_day_mm':'Найбільші добові опади, мм',
    'max_temp_c':'Найвища добова Tmax, °C',
    'dry_spell_days':'Найдовший сухий період, днів',
}

def longest_boolean_run(mask: pd.Series) -> int:
    largest = run = 0
    for val in mask.fillna(False):
        if bool(val):
            run += 1
            largest = max(largest, run)
        else:
            run = 0
    return largest


def segment_indicators(g: pd.DataFrame) -> dict:
    rain, tmax = g['rain_mm'], g['tmax_c']
    return {
        'rain_mm': float(rain.sum(min_count=1)),
        'dry_days': int((rain < 1).sum()),
        'heavy_rain_days': int((rain >= 20).sum()),
        'hot_days': int((tmax >= 30).sum()),
        'wettest_day_mm': float(rain.max()),
        'max_temp_c': float(tmax.max()),
        'dry_spell_days': int(longest_run_of_dry_days(rain)),
        'coldest_night_c': float(g['tmin_c'].min()),
        'p95_wet_day_mm': float(rain[rain >= 1].quantile(.95)) if (rain >= 1).sum() >= 5 else np.nan,
    }


def aggregate_monthly(daily: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
    rows = []
    for (year, month), g in daily.groupby([daily.index.year, daily.index.month]):
        expected = calendar.monthrange(int(year), int(month))[1]
        cov = min(g[col].count() / expected for col in ('rain_mm','tmax_c','tmin_c'))
        if cov < min_coverage:
            continue
        rows.append(dict(year=int(year), month=int(month), month_name=MONTHS_UK[int(month)],
                         coverage=round(cov,4), **segment_indicators(g)))
    return pd.DataFrame(rows)


def aggregate_seasonal(daily: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
    rows = []
    df = daily.copy()
    mo = df.index.month
    df['season'] = np.select([mo.isin(SEASON_MONTHS[k]) for k in SEASON_MONTHS],
                             list(SEASON_MONTHS), default='?')
    df['season_year'] = df.index.year + (mo == 12).astype(int)
    for (yr, season), g in df.groupby(['season_year','season']):
        if season not in SEASON_MONTHS:
            continue
        # DJF year N: Dec (N-1) + Jan/Feb (N).
        start = pd.Timestamp(year=int(yr)-1, month=12, day=1) if season=='DJF' else pd.Timestamp(year=int(yr), month=SEASON_MONTHS[season][0], day=1)
        end = (start + pd.DateOffset(months=3)) - pd.Timedelta(days=1)
        expected = len(pd.date_range(start,end,freq='D'))
        coverage = min(g[col].count()/expected for col in ('rain_mm','tmax_c','tmin_c'))
        if coverage < min_coverage or len(g.index.to_period('M').unique()) < 3:
            continue
        rows.append(dict(year=int(yr), season=season, season_name=SEASONS_UK[season],
                         coverage=round(coverage,4), **segment_indicators(g)))
    return pd.DataFrame(rows)


def fdr_bh(pvalues) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values; preserve NaNs."""
    p = np.asarray(list(pvalues),dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    ranks = np.arange(1,len(order)+1)
    raw = p[order]*len(order)/ranks
    out[order] = np.clip(np.minimum.accumulate(raw[::-1])[::-1],0,1)
    return out


def segmented_trends(monthly: pd.DataFrame, seasonal: pd.DataFrame, field: str, kind: str) -> pd.DataFrame:
    data = monthly if kind=='month' else seasonal
    key = 'month' if kind=='month' else 'season'
    if data.empty:
        return pd.DataFrame()
    rows = []
    groups = range(1,13) if kind=='month' else ('DJF','MAM','JJA','SON')
    for group in groups:
        one = data[data[key] == group]
        r = calc_trend(one, field)
        rows.append({'Період': MONTHS_UK[group] if kind=='month' else SEASONS_UK[group],
                     'year_count':r['n'], 'Зміна/10 років':r['slope_decade'],
                     'p-value':r['p'], 'Нижня ДІ95':r['lower_decade'],
                     'Верхня ДІ95':r['upper_decade']})
    result = pd.DataFrame(rows)
    result['p_FDR'] = fdr_bh(result['p-value'])
    result['FDR<0.05'] = result['p_FDR'].lt(.05) & result['p_FDR'].notna()
    return result


def contiguous_events(daily: pd.DataFrame, col: str, threshold: float, min_days=3, less=True) -> pd.DataFrame:
    """Exact runs; NaN and nonconsecutive timestamps break streaks."""
    s = daily[col].sort_index()
    rows = []
    start = last = None
    count = 0
    def finish():
        if count >= min_days and start is not None:
            rows.append({'Початок':start.date().isoformat(), 'Кінець':last.date().isoformat(),
                         'Тривалість, днів':count})
    for dt, val in s.items():
        qualifying = bool(pd.notna(val) and (val < threshold if less else val >= threshold))
        consecutive = last is not None and dt - last == pd.Timedelta(days=1)
        if not qualifying or (start is not None and not consecutive):
            finish()
            start = None
            count = 0
        if qualifying:
            if start is None:
                start=dt
            count+=1
            last=dt
        else:
            last=None
    finish()
    return pd.DataFrame(rows,columns=['Початок','Кінець','Тривалість, днів']).sort_values('Тривалість, днів',ascending=False,ignore_index=True)


def plot_daily(daily: pd.DataFrame, value_col: str, period_year: int):
    g = daily[daily.index.year==period_year]
    pretty = {'rain_mm':'Щоденні опади, мм','tmax_c':'Добова максимальна температура, °C',
              'tmin_c':'Добова мінімальна температура, °C'}
    fig=go.Figure()
    if value_col=='rain_mm':
        fig.add_bar(x=g.index,y=g[value_col],name=pretty[value_col],marker_color='#38bdf8')
    else:
        fig.add_scatter(x=g.index,y=g[value_col],mode='lines',name=pretty[value_col],line={'color':'#f97316' if value_col=='tmax_c' else '#38bdf8','width':1.8})
    fig.update_layout(title=f'{pretty[value_col]} — {period_year}',height=400,
                      xaxis_title='Дата',yaxis_title=pretty[value_col],margin=dict(t=55,b=40,l=40,r=20))
    return fig


def plot_monthly_heatmap(monthly: pd.DataFrame, field: str):
    pivot = monthly.pivot(index='month',columns='year',values=field).reindex(range(1,13))
    fig = go.Figure(go.Heatmap(x=pivot.columns,y=[MONTHS_UK[i] for i in pivot.index],
                               z=pivot.values,colorscale='Viridis',
                               colorbar=dict(title=SEG_FIELDS[field]),
                               hovertemplate='Рік %{x}<br>%{y}<br>Значення %{z:.1f}<extra></extra>'))
    fig.update_layout(title=f'Календарна мінливість: {SEG_FIELDS[field]}',
                      height=490,margin=dict(t=60,b=35,l=70,r=15), xaxis_title='Рік')
    return fig


def spatial_points(lat: float,lon: float, delta: float=0.75) -> list[tuple[str,float,float]]:
    return [('Центр',lat,lon),('Північ',min(lat+delta,89.8),lon),
            ('Південь',max(lat-delta,-89.8),lon),
            ('Захід',lat,(lon-delta+180)%360-180),('Схід',lat,(lon+delta+180)%360-180)]


def spatial_month_data(site: dict, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, per_day = [],[]
    for name, lat, lon in spatial_points(site['lat'],site['lon']):
        # Reuse previously retrieved daily series for the center. Neighbors only 1 yr.
        if name=='Центр':
            data=site['daily']
        else:
            data=fetch_nasa_power(lat,lon,year,year)
        g=data[(data.index.year==year)&(data.index.month==month)]
        if g.empty:
            continue
        cover=min(g[x].count()/calendar.monthrange(year,month)[1] for x in ('rain_mm','tmax_c'))
        if cover < .9:
            continue
        z=segment_indicators(g)
        records.append({'Точка':name,'lat':lat,'lon':lon,'Покриття':round(cover,3),
                        'Опади, мм':z['rain_mm'],'Сухих днів':z['dry_days'],
                        'Сильних дощів':z['heavy_rain_days'],
                        'Спекотних днів':z['hot_days'],
                        'Максимум Tmax, °C':z['max_temp_c'],
                        'Максимум опадів за добу, мм':z['wettest_day_mm']})
        for dt,row in g.iterrows():
            per_day.append({'Дата':dt,'Точка':name,'Опади, мм/добу':row['rain_mm'],
                            'Tmax, °C':row['tmax_c'],'Tmin, °C':row['tmin_c']})
    return pd.DataFrame(records),pd.DataFrame(per_day)


# All research sections are visible on the same page; navigation uses real page anchors.
ORION_SECTIONS = [
    ("orion-day", "Кожен день", "01", "🗓️"),
    ("orion-month", "Місяці та сезони", "02", "📅"),
    ("orion-space", "Просторова різниця", "03", "📍"),
    ("orion-trends", "Багаторічні тренди", "04", "📈"),
    ("orion-regions", "Порівняння регіонів", "05", "🗺️"),
    ("orion-ai", "AI — аномалії", "06", "🧠"),
    ("orion-actions", "Практичні висновки", "07", "🌱"),
    ("orion-methods", "Дані й методи", "08", "📥"),
]


def section_navigation(sidebar: bool = False) -> str:
    links = ''.join(
        f'<a href="#{slug}">{emoji} {number} · {title}</a>'
        for slug, title, number, emoji in ORION_SECTIONS
    )
    cls = 'orion-side-nav' if sidebar else 'orion-nav'
    return f'<nav class="{cls}" aria-label="Розділи ORION Earth">{links}</nav>'


def section_heading(slug: str, label: str, number: str) -> None:
    # A real HTML id enables jumps from both the sidebar and page navigation.
    st.markdown(
        f'<div id="{slug}" class="orion-section-head">'
        f'<span>ORION EARTH · РОЗДІЛ {number} / 08</span><h2>{label}</h2></div>',
        unsafe_allow_html=True,
    )


# ─────────────────── STREAMLIT APPLICATION ───────────────────
def render_offline_showcase(sites: list[dict]) -> str:
    """Create a no-network, Plotly-powered HTML report ONLY from already retrieved NASA data."""
    if not sites:
        raise ValueError("Немає отриманих з NASA даних для HTML-звіту")
    css = """
    *{box-sizing:border-box} body{margin:0;background:#080f1c;color:#e9f3ff;font:16px system-ui,Segoe UI,Arial,sans-serif}
    main{max-width:1450px;margin:auto;padding:28px 28px 70px}header{padding:24px 0 20px;border-bottom:1px solid #25384f}
    .eyebrow{font-size:12px;font-weight:800;letter-spacing:.18em;color:#49d2ef}h1{font-size:clamp(28px,4vw,48px);margin:10px 0}
    h2{margin:34px 0 16px}.muted{color:#9db3c9}.pill{display:inline-block;border:1px solid #2e6584;background:#10273e;color:#6bdeec;border-radius:99px;padding:7px 12px;font-size:13px}
    .metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:26px 0}
    .metric{padding:20px;background:linear-gradient(135deg,#102740,#102036);border:1px solid #243f58;border-radius:14px}
    .metric small{display:block;color:#a5bcd1}.metric strong{font-size:26px;display:block;margin-top:10px}
    .figure{background:#101b2b;border:1px solid #203a50;padding:5px;border-radius:15px;margin:14px 0}
    details{border-top:1px solid #264257;padding:15px 0}summary{cursor:pointer;font-size:19px;font-weight:700}
    .note{padding:14px 20px;background:#102539;border-left:3px solid #45d4e7;border-radius:8px;margin:20px 0}
    .warn{border-left-color:#e0a06b}.foot{border-top:1px solid #213750;margin-top:36px;padding-top:18px;font-size:13px;color:#a2b6c9}
    a{color:#70d9f7} @media(max-width:700px){main{padding:12px}}
    """
    plotly_lib_written = False
    sections=[]
    for site in sites:
        n=html.escape(str(site['name']))
        df=site['daily'].copy()
        annual=site['annual'].copy()
        latest=int(annual['year'].max())
        yr=df[df.index.year==latest]
        t=site['trends']
        meta=read_cache_record(site['lat'], site['lon'], int(df.index.year.min()),int(df.index.year.max()))
        if not meta:
            raise RuntimeError(f"Відсутня копія первинних даних NASA для {site['name']}; неможливо створити офлайн-звіт")
        source_url=html.escape(str(meta['api_url']),quote=True)
        time=html.escape(str(meta['retrieved_at_utc']))
        fig_daily=go.Figure()
        fig_daily.add_trace(go.Bar(x=yr.index,y=yr['rain_mm'],name='Опади, мм/добу',marker_color='#4fd3f2'))
        fig_daily.add_trace(go.Scatter(x=yr.index,y=yr['tmax_c'],name='Tmax, °C',yaxis='y2',line=dict(color='#ffa25c',width=2)))
        fig_daily.update_layout(title=f'Кожна доба {latest}: опади і Tmax',template='plotly_dark',paper_bgcolor='#101b2b',plot_bgcolor='#101b2b',font_color='#cbdff0',height=390,margin=dict(t=55,l=40,r=45,b=45),
            yaxis=dict(title='Опади, мм/день'),yaxis2=dict(title='Tmax, °C',overlaying='y',side='right'),legend=dict(orientation='h',y=-.22))
        fig_tem=trend_plot(site,'mean_temp_c')
        fig_tem.update_layout(paper_bgcolor='#101b2b',plot_bgcolor='#101b2b',font_color='#cbdff0',template='plotly_dark')
        fig_prec=trend_plot(site,'rain_mm')
        fig_prec.update_layout(paper_bgcolor='#101b2b',plot_bgcolor='#101b2b',font_color='#cbdff0',template='plotly_dark')
        plot_html=[]
        for fig in (fig_daily,fig_tem,fig_prec):
            plot_html.append(pio.to_html(fig,include_plotlyjs='inline' if not plotly_lib_written else False,
                                           full_html=False,config={'responsive':True,'displaylogo':False}))
            plotly_lib_written=True
        latest_year=annual[annual['year']==latest].iloc[0]
        r=t['mean_temp_c']
        val=f"{r['slope_decade']:+.2f} °C/10 р." if np.isfinite(r['slope_decade']) else 'Недостатньо даних'
        significance='FDR<0,05' if r['significant_fdr'] else 'не підтверджено після FDR'
        sections.append(f"""
        <details open><summary>📍 {n} · {site['lat']:.4f}, {site['lon']:.4f}</summary>
        <p class='muted'>NASA POWER / UTC · завантажено: {time} · {len(df):,} календарних днів · <a href='{source_url}'>Вихідний API-запит</a></p>
        <div class='metrics'>
          <div class='metric'><small>Рік добового прикладу</small><strong>{latest}</strong></div>
          <div class='metric'><small>Максимальна добова сума опадів, весь період</small><strong>{df['rain_mm'].max():.1f} мм</strong></div>
          <div class='metric'><small>Спекотні й сухі дні, {latest}</small><strong>{int(latest_year['hot_and_dry_days'])}</strong></div>
          <div class='metric'><small>Температурний тренд</small><strong>{val}</strong><small>{significance}</small></div>
        </div>
        <div class='figure'>{plot_html[0]}</div>
        <div class='figure'>{plot_html[1]}</div>
        <div class='figure'>{plot_html[2]}</div>
        <div class='note'>Це знімок реальних сіткових даних NASA POWER, збережений після успішного запиту. Інтернет для перегляду цього файлу не потрібний. Сама модель не прогнозує конкретні природні лиха.</div>
        </details>""")
    page=f"""<!doctype html><html lang='uk'><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>
    <title>ORION Earth · NASA Data Showcase</title><style>{css}</style></head><body><main>
    <header><span class='eyebrow'>ORION / NASA POWER · EVIDENCE-BASED CLIMATE INTELLIGENCE</span>
    <h1>🌍 ORION EARTH</h1><span class='pill'>✓ Лише справжні дані NASA POWER</span>
    <p class='muted'>Інтерактивний офлайн-звіт. Первинні щоденні ряди отримані з NASA POWER API. Жодних вигаданих даних.</p></header>
    <h2>Докази змін клімату: добові спостереження і тенденції</h2>
    {''.join(sections)}
    <div class='note warn'>Важливо: дані NASA POWER — це просторово усереднені сіткові метеорологічні продукти. Не підміняють показники метеостанцій, локальних гідрологічних моделей або систем оперативного попередження.</div>
    <div class='foot'>ORION Earth · Незалежний науковий прототип · Джерело даних: <a href='https://power.larc.nasa.gov/'>NASA POWER</a> · NASA не є розробником цього застосунку.</div>
    </main></body></html>"""
    return page


st.markdown('''<div style="display:flex;align-items:center;gap:12px;margin-bottom:7px"><span style="background:#17334c;padding:8px 12px;border-radius:13px;font-size:1.5rem">🌐</span><span style="font-size:.78rem;font-weight:800;letter-spacing:.17em;color:#56cdef">ORION / EARTH SCIENCE INTELLIGENCE</span></div>''', unsafe_allow_html=True)
st.title('ORION EARTH')
st.markdown('<div style="color:#a8c9dc;font-size:1.03rem;margin-top:-.5rem;margin-bottom:.9rem">Earth intelligence · Доказовий аналіз кліматичних змін із NASA POWER</div>',unsafe_allow_html=True)
st.caption('ORION Earth Web 5.0 · 8 відкритих розділів · NASA POWER / UTC · реальні добові дані · наукова відтворюваність')
st.caption('NASA POWER Daily API • щоденні значення • екстремуми • місяці та сезони • просторова неоднорідність • AI')
st.write('**Головна ідея:** виявляти конкретні кліматичні зміни та їхні можливі наслідки, не приховуючи відмінностей усередині річних середніх величин.')
st.markdown('### 🧭 Перехід до розділів')
st.markdown(section_navigation(), unsafe_allow_html=True)
st.caption('Розділи показуються один під одним; кнопки вище та в лівому меню переводять до відповідної частини сторінки.')

with st.sidebar:
    st.header('⚙️ Дослідження')
    st.markdown('### 🧭 Усі розділи')
    st.markdown(section_navigation(sidebar=True), unsafe_allow_html=True)
    st.caption('Усі 8 розділів розміщені на одній сторінці, без прихованих вкладок.')
    st.divider()
    choices=st.multiselect('Координатні точки (1–5)',list(LOCATIONS),
                           default=['Дрогобич, Україна'],max_selections=5)
    custom=st.checkbox('Додати власну точку')
    if custom:
        custom_name=st.text_input('Назва точки','Моя точка')
        lat=st.number_input('Широта',min_value=-89.8,max_value=89.8,value=49.35,format='%.4f')
        lon=st.number_input('Довгота',min_value=-180.,max_value=180.,value=23.50,format='%.4f')
    period=st.slider('Період, повні роки',min_value=1982,max_value=2025,value=(2001,2025))
    st.caption('Для статистичних тестів потрібно ≥10 років. Онлайн-сервер повторно використовує кеш NASA, поки він доступний.')
    run=st.button('🚀 Аналізувати реальні дані NASA',type='primary',use_container_width=True)
    demo_run=st.button('⚡ Швидкий показ: Дрогобич ↔ Афіни',use_container_width=True,help='Завантажує справжні NASA POWER за 2010–2025 роки для двох точок. Не тестові дані.')
    st.markdown('---')
    st.caption('Живі запити виконуються до офіційного NASA POWER API. Кеш на хостингу тимчасовий; після перезапуску сервера може знадобитися новий запит.')

if run or demo_run:
    inputs=[(n,*LOCATIONS[n]) for n in choices] if not demo_run else [(n,*LOCATIONS[n]) for n in ('Дрогобич, Україна','Афіни, Греція')]
    if demo_run:
        period=(2010,2025)
    if custom and not demo_run:
        if not custom_name.strip():
            st.error('Потрібна назва власної точки.'); st.stop()
        if any(n==custom_name.strip() for n,*_ in inputs):
            st.error('Назва має бути унікальною.');st.stop()
        inputs.append((custom_name.strip(),float(lat),float(lon)))
    if not inputs:
        st.error('Вибери хоча б одну точку.');st.stop()
    if len(inputs) > 5:
        st.error('Можна аналізувати щонайбільше 5 точок за один запит.'); st.stop()
    if period[1]-period[0]+1<10:
        st.error('Обери ≥10 років для статистичного аналізу.');st.stop()
    sites,errors=[],[]
    prog=st.progress(0.,text='Завантаження NASA POWER...')
    for i,(name,lat,lon) in enumerate(inputs):
        try:
            sites.append(analyze_site(name,lat,lon,period[0],period[1]))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
        prog.progress((i+1)/len(inputs),text=f'{i+1}/{len(inputs)} точок')
    prog.empty()
    for err in errors:
        st.error(err)
    if not sites:
        st.stop()
    st.session_state['orion_sites_v2']=sites
    st.session_state['orion_period_v2']=period
    st.session_state['orion_spatial_v2']={}
    st.session_state.pop('orion_showcase_html', None)
    try:
        st.session_state['orion_showcase_html'] = render_offline_showcase(sites)
        st.success('✓ Дані NASA завантажено й проаналізовано. HTML-презентацію можна завантажити в розділі «Дані й методи».')
    except Exception as exc:
        st.warning(f'Аналіз доступний, але автономний HTML-звіт не вдалося створити: {exc}')

if 'orion_sites_v2' not in st.session_state:
    st.info('Вибери координатні точки й натисни «Аналізувати реальні дані NASA» або «Швидкий показ». Без NASA-запиту чи перевіреного кешу застосунок не показуватиме кліматичних чисел.')
    st.markdown('**Усі вісім розділів доступні для перегляду.** Для кліматичних графіків та цифр спершу виконай NASA-запит; жодні дані не генеруються автоматично.')
    for slug, title, number, emoji in ORION_SECTIONS:
        section_heading(slug, f'{emoji} {title}', number)
        if slug == 'orion-methods':
            st.markdown('**Джерело:** [NASA POWER Daily Point API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/). **Добові змінні:** PRECTOTCORR, T2M, T2M_MAX, T2M_MIN; UTC.')
            st.markdown('**Методи:** Kendall tau, нахил Theil–Sen, Benjamini–Hochberg FDR; порогові індикатори сухих, дощових і спекотних днів. **AI:** Isolation Forest для незвичайних комбінацій показників.')
            st.caption('Реальні результати, API URL, SHA-256 джерела та файли CSV/JSON/HTML з’являться після завантаження NASA.')
        else:
            st.info('Щоб побачити цей розділ із даними, обери координати й натисни «Аналізувати реальні дані NASA» в лівій панелі.')
    st.stop()

sites=st.session_state['orion_sites_v2']
if st.session_state.get('orion_showcase_html'):
    st.info('📊 HTML-звіт зі справжніх даних NASA готовий для завантаження в розділі «Дані й методи».')
else:
    st.info('HTML-звіт з’явиться після успішного аналізу даних NASA.')
active_name=st.selectbox('📍 Детальний аналіз точки',[s['name'] for s in sites])
site=next(s for s in sites if s['name']==active_name)
daily=site['daily']
annual=site['annual']
monthly=aggregate_monthly(daily)
seasonal=aggregate_seasonal(daily)
t=site['trends']

c1,c2,c3,c4=st.columns(4)
c1.metric('Днів із повними даними',f"{int(daily[['rain_mm','temp_c','tmax_c','tmin_c']].notna().all(axis=1).sum()):,}".replace(',',' '))
c2.metric('Найсильніший добовий дощ',f"{daily['rain_mm'].max():.1f} мм")
c3.metric('Найвища Tmax',f"{daily['tmax_c'].max():.1f} °C")
c4.metric('Тренд середньорічної T',f"{t['mean_temp_c']['slope_decade']:+.2f} °C/10 р.",
          'FDR<0,05' if t['mean_temp_c']['significant_fdr'] else 'не підтверджено після FDR')
st.caption('Угорі: добові екстремуми (не середні), окремо — багаторічний температурний тренд. Дані NASA POWER мають сіткову роздільність.')
if not annual.empty:
    latest_stats=annual.iloc[-1]
    c5,c6=st.columns(2)
    c5.metric(f"Спекотні й сухі дні у {int(latest_stats['year'])}",int(latest_stats['hot_and_dry_days']))
    c6.metric(f"Зливи після ≥7 сухих днів у {int(latest_stats['year'])}",int(latest_stats['dry_then_heavy_rain']))


section_heading('orion-day', '🗓️ Кожен день', '01')
with st.container():
    st.subheader('Щоденні, неусереднені дані NASA')
    col_year,col_var=st.columns([1,2])
    with col_year:
        selected_year=st.selectbox('Рік',sorted(daily.index.year.unique(),reverse=True),key='daily_year')
    with col_var:
        selected_var=st.selectbox('Щоденний показник', ['rain_mm','tmax_c','tmin_c'],
            format_func=lambda k:{'rain_mm':'Опади за кожну добу','tmax_c':'Максимальна температура кожної доби',
                                  'tmin_c':'Мінімальна температура кожної доби'}[k])
    st.plotly_chart(plot_daily(daily,selected_var,int(selected_year)),use_container_width=True)
    filtered=daily[daily.index.year==selected_year]
    a,b,c,d=st.columns(4)
    a.metric('Максимум опадів за добу',f"{filtered['rain_mm'].max():.1f} мм")
    b.metric('Днів із ≥20 мм',str(int((filtered['rain_mm']>=20).sum())))
    c.metric('Сухих днів (<1 мм)',str(int((filtered['rain_mm']<1).sum())))
    d.metric('Днів ≥30°C',str(int((filtered['tmax_c']>=30).sum())))
    st.markdown('#### Послідовні події — конкретні дати')
    dry_events=contiguous_events(filtered,'rain_mm',1,min_days=3,less=True)
    hot_events=contiguous_events(filtered,'tmax_c',30,min_days=3,less=False)
    ed1,ed2=st.columns(2)
    with ed1:
        st.write('**Сухі періоди (≥3 доби)**')
        if dry_events.empty: st.caption('Не знайдено.')
        else: st.dataframe(dry_events.head(30),hide_index=True,use_container_width=True)
    with ed2:
        st.write('**Спекотні серії Tmax ≥30°C (≥3 доби)**')
        if hot_events.empty: st.caption('Не знайдено.')
        else: st.dataframe(hot_events.head(30),hide_index=True,use_container_width=True)
    st.write('**Десять днів із найсильнішими опадами**')
    peaks=filtered[['rain_mm','tmax_c','tmin_c']].nlargest(10,'rain_mm').reset_index()
    peaks.columns=['Дата','Опади, мм','Tmax, °C','Tmin, °C']
    st.dataframe(peaks,hide_index=True,use_container_width=True)
    st.download_button('⬇️ Всі добові записи — CSV',daily.to_csv(index=True).encode('utf-8-sig'),
                       file_name='orion_daily_nasa.csv',mime='text/csv')

section_heading('orion-month', '📅 Місяці та сезони', '02')
with st.container():
    st.subheader('Диференціація за календарними місяцями і сезонами')
    if monthly.empty:
        st.warning('Недостатньо повних місяців для аналізу. Інші розділи залишаються доступними.')
    if not monthly.empty:
        metric=st.selectbox('Показник для місячної теплової карти',list(SEG_FIELDS),format_func=lambda k:SEG_FIELDS[k])
        st.plotly_chart(plot_monthly_heatmap(monthly,metric),use_container_width=True)
        st.caption('Кожна клітинка — свій місяць свого року. Сума опадів — накопичення за місяць, решта — кількість подій або максимум конкретної доби, а не усереднення за десятиліття.')
        st.markdown('#### Зміни для кожного місяця окремо')
        st.info('Для 12 місячних тестів одного показника p-value скориговано методом Benjamini–Hochberg (FDR). Це зменшує ризик хибнопозитивних знахідок.')
        st.dataframe(segmented_trends(monthly,seasonal,metric,'month'),hide_index=True,use_container_width=True)
        st.markdown('#### Сезони (зима: грудень попереднього року + січень і лютий поточного)')
        s1,s2=st.columns([1,2])
        with s1:
            selected_season=st.selectbox('Сезон',list(SEASONS_UK),format_func=lambda x:SEASONS_UK[x])
        with s2:
            season_years=seasonal.loc[seasonal['season']==selected_season].sort_values('year')
            if not season_years.empty:
                fig=px.bar(season_years,x='year',y=metric,title=f'{SEASONS_UK[selected_season]}: {SEG_FIELDS[metric]}')
                st.plotly_chart(fig,use_container_width=True)
        st.dataframe(segmented_trends(monthly,seasonal,metric,'season'),hide_index=True,use_container_width=True)
        st.caption('Наявність статистичного тренду не означає, що знайдено причину. p-value не враховують автоматично часову автокореляцію; результати попередні.')

section_heading('orion-space', '📍 Просторова різниця', '03')
with st.container():
    st.subheader('Просторова диференціація — не одне число для всієї території')
    st.write('Для вибраного місяця завантажуються **окремі добові ряди для 5 точок**: центр, північ, південь, захід і схід (відступ 0,75°).')
    st.warning('Роздільність NASA POWER обмежена кліматичною сіткою. Різні адреси всередині однієї комірки можуть давати однакові значення. Цей модуль **не є мікрокліматичною картою вулиць**.')
    sp1,sp2=st.columns(2)
    with sp1:
        map_year=int(st.selectbox('Рік просторового аналізу',sorted(daily.index.year.unique(),reverse=True),key='map_year'))
    with sp2:
        map_month=int(st.selectbox('Місяць просторового аналізу',list(range(1,13)),
                                    format_func=lambda m:MONTHS_UK[m],key='map_month'))
    spatial_key=(site['name'],map_year,map_month)
    spatial_storage=st.session_state['orion_spatial_v2']
    if st.button('🌐 Отримати 5 координатних рядів NASA',type='primary'):
        try:
            with st.spinner('NASA POWER: завантаження чотирьох сусідніх точок...'):
                spatial_storage[spatial_key]=spatial_month_data(site,map_year,map_month)
        except Exception as exc:
            st.error(f'Не вдалося завантажити сусідні точки: {exc}')
    if spatial_key in spatial_storage:
        spatial_table,spatial_daily=spatial_storage[spatial_key]
        if not spatial_table.empty:
            spatial_metric=st.selectbox('Показник для карти',
              ['Опади, мм','Сухих днів','Сильних дощів','Спекотних днів','Максимум Tmax, °C','Максимум опадів за добу, мм'])
            map_fig=px.scatter_map(spatial_table,lat='lat',lon='lon',color=spatial_metric,
                         size_max=19,hover_name='Точка',size=[18]*len(spatial_table),
                         hover_data={'lat':':.3f','lon':':.3f','Покриття':True,spatial_metric:':.1f'},
                         map_style='open-street-map',zoom=6,height=470)
            map_fig.update_layout(margin=dict(t=0,r=0,b=0,l=0))
            st.plotly_chart(map_fig,use_container_width=True)
            st.dataframe(spatial_table,hide_index=True,use_container_width=True)
            sel_date=st.selectbox('Порівняння конкретної доби',
                                  sorted(spatial_daily['Дата'].dropna().unique()),format_func=lambda d:str(pd.Timestamp(d).date()))
            snapshot=spatial_daily[spatial_daily['Дата']==sel_date]
            st.dataframe(snapshot[['Точка','Опади, мм/добу','Tmax, °C','Tmin, °C']],hide_index=True,use_container_width=True)
            st.download_button('⬇️ Просторова добова таблиця — CSV',
                   spatial_daily.to_csv(index=False).encode('utf-8-sig'),
                   file_name=f'orion_spatial_{map_year}_{map_month:02d}.csv',mime='text/csv')
    else:
        st.info('Натисни кнопку, щоб отримати окремі ряди сусідніх комірок. Запити виконуються лише за твоєю командою, а не автоматично на кожній зміні графіка.')

section_heading('orion-trends', '📈 Багаторічні тренди', '04')
with st.container():
    st.subheader('Довгострокові тренди — додатковий рівень узагальнення')
    left,right=st.columns(2)
    with left:
        st.plotly_chart(trend_plot(site,'rain_mm'),use_container_width=True)
        st.caption(label_trend(t['rain_mm'],'мм/10 років'))
    with right:
        st.plotly_chart(trend_plot(site,'dry_spell_days'),use_container_width=True)
        st.caption(label_trend(t['dry_spell_days'],'днів/10 років'))
    extra=st.selectbox('Додаткові річні індикатори',list(METRICS),index=3,format_func=lambda k:METRICS[k][0])
    st.plotly_chart(trend_plot(site,extra),use_container_width=True)
    st.caption(label_trend(t[extra],METRICS[extra][2]))
    summary=[]
    for field,(name,unit,rate_unit) in METRICS.items():
        r=t[field]
        summary.append({'Показник':name,'Нахил за 10 років':r['slope_decade'],
                        'p (Kendall)':r['p'], 'p_FDR (14 тестів)':r['p_fdr'],
                        'ДІ95 нижня':r['lower_decade'],'ДІ95 верхня':r['upper_decade'],
                        'Значущий після FDR':r['significant_fdr']})
    st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True)
    if site['paradox_confirmed']:
        st.success('За попереднім критерієм є одночасний статистично значущий ріст опадів та найдовшого сухого періоду.')
    elif site['paradox_candidate']:
        st.warning('Напрями обох трендів додатні, але статистичної значущості обох не підтверджено.')
    else:
        st.info('Парадокс «більше річних опадів і довший сухий період» тут не підтверджено.')

section_heading('orion-regions', '🗺️ Порівняння регіонів', '05')
with st.container():
    df_reg=pd.DataFrame([{'Регіон':s['name'],'lat':s['lat'],'lon':s['lon'],
                          'Парадокс':'Підтверджено' if s['paradox_confirmed'] else 'Гіпотеза' if s['paradox_candidate'] else 'Не виявлено'}
                         for s in sites])
    figmap=px.scatter_map(df_reg,lat='lat',lon='lon',hover_name='Регіон',color='Парадокс',
                          map_style='open-street-map',zoom=1 if len(sites)>1 else 5,height=430)
    figmap.update_traces(marker=dict(size=14))
    figmap.update_layout(margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(figmap,use_container_width=True)
    sel_metric_1,sel_metric_2=st.columns(2)
    with sel_metric_1:
        metric_1=st.selectbox('Показник А',list(METRICS),index=0,
                   format_func=lambda k:METRICS[k][0])
    with sel_metric_2:
        metric_2=st.selectbox('Показник Б',list(METRICS),index=4,
                   format_func=lambda k:METRICS[k][0])
    l,r=st.columns(2)
    with l: st.plotly_chart(make_comparison(sites,metric_1),use_container_width=True)
    with r: st.plotly_chart(make_comparison(sites,metric_2),use_container_width=True)
    st.caption('Це порівняння окремих координатних комірок NASA POWER. Немає підстав узагальнювати їх на всю країну.')

section_heading('orion-ai', '🧠 AI — аномалії', '06')
with st.container():
    st.subheader('🧠 Аномальні дні — незалежно від річних середніх')
    st.write('Порівнюємо конкретні щоденні опади, Tmax і Tmin з типовими значеннями **відповідного місяця**. '
             'Isolation Forest виділяє 2% незвичайних комбінацій; додатково перевіряються добові опади й температури за крайніми 0,5% розподілів кожного місяця.')
    with st.spinner('Пошук нетипових днів...'):
        daily_anomalies = detect_daily_anomalies(daily)
    st.metric('Позначено незвичайних днів',len(daily_anomalies))
    if not daily_anomalies.empty:
        st.dataframe(daily_anomalies.head(60),hide_index=True,use_container_width=True)
        st.download_button('⬇️ Дати й характеристики добових аномалій — CSV',
           daily_anomalies.to_csv(index=False).encode('utf-8-sig'),
           file_name='orion_daily_anomalies.csv',mime='text/csv')
    st.caption('Це статистичні незвичайності без підтвердження небезпеки. '
               'Не кожен незвичайний день — природне лихо; і не кожне лихо має бути тут позначене.')
    st.divider()
    st.subheader('Порівняння з AI на річних індексах (історичний модуль)')
    flagged=annual[annual['anomaly']]
    if annual['anomaly_score'].notna().any():
        st.write(f'Isolation Forest позначив {len(flagged)} із {len(annual)} років як нетипові (евристика, contamination=0,15).')
        d=annual.copy();d['Група']=np.where(d['anomaly'],'Нетиповий','Звичайний')
        fig=px.scatter(d,x='rain_mm',y='dry_spell_days',color='Група',hover_data=['year','mean_temp_c','hot_days'],
                       labels={'rain_mm':'Опади за рік, мм','dry_spell_days':'Найдовший сухий період, днів'})
        st.plotly_chart(fig,use_container_width=True)
        if len(flagged):
            st.dataframe(flagged[['year','rain_mm','dry_spell_days','hot_days','anomaly_score']],hide_index=True,use_container_width=True)
    else:
        st.warning('Для AI потрібно щонайменше 12 повних років.')
    st.caption('Позначка AI — це нетипове поєднання річних показників, не прогноз катастрофи чи підтвердження загрози.')

section_heading('orion-actions', '🌱 Практичні висновки', '07')
with st.container():
    st.subheader('Практична цінність — висновки з даних, а не припущення')
    for header,desc in implications(site):
        st.markdown(f'**{header}**')
        st.write(desc)
    st.markdown('#### Що змінилося між початком і кінцем періоду')
    st.caption('Порівнюємо перші та останні 5 повних років. Це фактичні значення із NASA POWER, а не модельний прогноз. Різниця між періодами не є автоматично статистично значущою.')
    st.dataframe(evidence_comparison(site), hide_index=True, use_container_width=True)
    st.markdown('#### Панель пріоритетів для дослідника')
    alerts=[]
    if t['mean_temp_c']['significant_fdr']:
        alerts.append(('🌡️ Зміна теплових умов',
            f"Температура: {t['mean_temp_c']['slope_decade']:+.2f} °C за 10 років, "
            f"p={t['mean_temp_c']['p']:.3g}. Перевірити хвилі спеки, уразливість населення та вологість."))
    if t['heavy_rain_days']['significant_fdr']:
        alerts.append(('🌧️ Перевірка зливового навантаження',
            f"Частота днів із ≥20 мм: {t['heavy_rain_days']['slope_decade']:+.2f} "
            f"днів/10 років, p={t['heavy_rain_days']['p']:.3g}. Потрібна локальна гідрологічна оцінка."))
    if t['dry_spell_days']['significant_fdr']:
        alerts.append(('💧 Перевірка посушливих періодів',
            f"Найдовша суха серія: {t['dry_spell_days']['slope_decade']:+.2f} "
            f"днів/10 років, p={t['dry_spell_days']['p']:.3g}. Перевірити вологість ґрунту."))
    if alerts:
        for heading,detail in alerts:
            with st.container(border=True):
                st.markdown(f'**{heading}**')
                st.write(detail)
    else:
        st.info('За заданими індикаторами значущих змін не виявлено. Це не доводить відсутності кліматичних ризиків.')
    st.caption('Підставою для текстових сигналів слугує FDR-коригована значущість 14 річних індексів. Це дослідницькі сигнали, не оперативні попередження чи оцінки ймовірності лих.')
    st.markdown('#### Що перевіряти в деталях')
    st.write('**Агрономія:** конкретні дати та тривалість сухих періодів у сезон вирощування, максимум опадів у важливі фази розвитку рослин.')
    st.write('**Управління водою:** повторюваність добових злив і сухих днів у кожному місяці, а не тільки річна сума.')
    st.write('**Здоров’я:** сезонні серії спекотних днів і зміни добових максимумів; для ризиків потрібні також вологість, експозиція та дані про населення.')
    st.write('**Інфраструктура:** просторові відмінності сіткових осередків допомагають вибрати місця для детальнішого локального моніторингу.')
    st.warning('Без даних про ґрунти, населення, міську дренажну мережу й локальні спостереження система не рахує збитки, точні ймовірності лих чи персональний медичний ризик.')

section_heading('orion-methods', '📥 Дані й методи', '08')
with st.container():
    st.subheader('Відкриті дані, відтворюваність і обмеження')
    st.write('**Джерело:** NASA POWER Daily point API: `PRECTOTCORR` (мм/день), `T2M` (°C), `T2M_MAX` (°C), `T2M_MIN` (°C). Часова шкала UTC.')
    st.write('**Не усереднюємо первинні дані:** добові ряди зберігаються, індивідуальні дати екстремумів доступні; місячні суми й кількість подій обчислюються окремо. Середнє використовується лише там, де обраний показник прямо означає середню температуру.')
    st.write('**Контроль якості:** виключаємо −999/NaN і відновлюємо відсутні дати як NaN; місяць і сезон потребують ≥90% добових значень; рік ≥90%; зима охоплює грудень попереднього року.')
    st.write('**Статистика:** Kendall tau + Theil–Sen, поправка Benjamini–Hochberg (FDR) для 14 річних індексів та окремо 12 місяців / 4 сезонів кожного показника. Часова автокореляція не скоригована, отже результати залишаються дослідницькими.')
    st.write('**Простір:** NASA POWER надає дані на сітці з роздільністю, що залежить від вихідних продуктів. Точність адресної метеостанції не гарантується.')
    provenance=read_cache_record(site['lat'],site['lon'],int(daily.index.year.min()),int(daily.index.year.max()))
    if provenance:
        st.success(f"Підтверджене походження локальної копії: NASA POWER API; збережено {provenance['retrieved_at_utc']} UTC")
        st.code(provenance['api_url'])
        st.caption(f"SHA-256 вихідної JSON-відповіді: {provenance['payload_sha256']}")
    if st.session_state.get('orion_showcase_html'):
        st.download_button('⬇️ Автономний HTML-звіт із реальними даними NASA', st.session_state['orion_showcase_html'].encode('utf-8'),
                           file_name='ORION_SHOWCASE.html',mime='text/html')
    st.markdown('[Документація NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/)')
    export_buttons=[('Повні добові дані',daily,'orion_daily.csv'),('Місяці',monthly,'orion_monthly.csv'),
                    ('Сезони',seasonal,'orion_seasonal.csv'),('Річні індекси',annual,'orion_annual.csv')]
    for label,table,filename in export_buttons:
        st.download_button(f'⬇️ {label} — CSV',table.to_csv(index=(label=='Повні добові дані')).encode('utf-8-sig'),
                           file_name=filename,mime='text/csv',key='dl_'+filename)
    export={'project':'ORION Earth Differentiated','source':'NASA POWER Daily API',
            'coordinates':{'name':active_name,'lat':site['lat'],'lon':site['lon']},
            'years':[int(annual['year'].min()),int(annual['year'].max())],
            'parameters':PARAMETERS,'utc':True,'trends':site['trends'],
            'methods':'Kendall tau + Theil-Sen + BH FDR (14 annual metrics)',
            'limitations':'No autocorrelation correction; gridded NASA POWER data'}
    st.download_button('⬇️ Результати та метадані — JSON',
           data=json.dumps(export,ensure_ascii=False,indent=2),file_name='orion_metadata.json',mime='application/json')

st.caption('ORION Earth WEB 5.0 • Independent research prototype • Powered by NASA POWER data, not endorsed by NASA')
