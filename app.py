# app.py
"""
Lost Dog Prediction System - Complete Working Version
유기견 위치 예측 시스템 - Windows 실행 가능 버전
"""

import base64
import json
import math
import hashlib
from io import BytesIO
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np
import requests
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from PIL import Image

# ================= Flask =================
app = Flask(__name__)
CORS(app)

# 간단 캐시
cache_data: Dict[str, Tuple[dict, datetime]] = {}

def get_cache_key(lat, lon, params):
    key_str = f"{lat:.3f}_{lon:.3f}_{json.dumps(params, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()

# Overpass API 후보
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# ============== Geometry Utilities ==============
def small_bbox(lat: float, lon: float, km: float = 3.0) -> Tuple[float, float, float, float]:
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(lat)))
    return (lat - dlat/2, lon - dlon/2, lat + dlat/2, lon + dlon/2)

def create_grid(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W), dtype=np.float32)

def normalize01(grid: np.ndarray) -> np.ndarray:
    mn, mx = float(grid.min()), float(grid.max())
    if mx - mn < 1e-9:
        return np.zeros_like(grid, dtype=np.float32)
    return (grid - mn) / (mx - mn)

def blur(grid: np.ndarray, k: int = 1, iters: int = 1) -> np.ndarray:
    g = grid.copy()
    for _ in range(iters):
        tmp = np.zeros_like(g)
        for dc in range(-k, k + 1):
            tmp += np.roll(g, dc, axis=1)
        tmp /= (2 * k + 1)
        tmp2 = np.zeros_like(tmp)
        for dr in range(-k, k + 1):
            tmp2 += np.roll(tmp, dr, axis=0)
        g = tmp2 / (2 * k + 1)
    return g

def latlon_to_rc(lat: float, lon: float, bbox: Tuple, H: int, W: int) -> Tuple[int, int]:
    s, w, n, e = bbox
    r = int((lat - s) / (n - s) * (H - 1))
    c = int((lon - w) / (e - w) * (W - 1))
    return max(0, min(H-1, r)), max(0, min(W-1, c))

def rc_to_latlon(r: int, c: int, bbox: Tuple, H: int, W: int) -> Tuple[float, float]:
    s, w, n, e = bbox
    lat = s + (n - s) * (r / (H - 1))
    lon = w + (e - w) * (c / (W - 1))
    return float(lat), float(lon)

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ============== OSM Data Fetching ==============
def fetch_osm(bbox: Tuple, timeout_sec: float = 30.0) -> Dict:
    s, w, n, e = bbox
    bbox_ql = f"{s},{w},{n},{e}"
    query = f"""
    [out:json][timeout:{int(timeout_sec)}];
    (
      way["leisure"="park"]({bbox_ql});
      way["landuse"~"grass|forest"]({bbox_ql});
      way["natural"="water"]({bbox_ql});
      way["waterway"~"."]({bbox_ql});
      way["landuse"="residential"]({bbox_ql});
      way["landuse"="commercial"]({bbox_ql});
      way["highway"~"footway|path|cycleway|residential|living_street"]({bbox_ql});
      way["highway"~"trunk|primary|motorway"]({bbox_ql});
      way["railway"~"rail|subway|light_rail"]({bbox_ql});
    );
    out body; >; out skel qt;
    """.strip()

    headers = {"Content-Type": "text/plain"}
    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data=query.encode("utf-8"), headers=headers, timeout=timeout_sec)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue

    # fallback 간단 쿼리
    query2 = f"""
    [out:json][timeout:{int(timeout_sec)}];
    (
      way["leisure"="park"]({bbox_ql});
      way["landuse"~"grass|forest"]({bbox_ql});
      way["natural"="water"]({bbox_ql});
      way["highway"~"."]({bbox_ql});
    );
    out body; >; out skel qt;
    """.strip()

    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data=query2.encode("utf-8"), headers=headers, timeout=timeout_sec)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue

    return {"elements": []}

# ============== Rasterization ==============
def rasterize(elements: List[Dict], bbox: Tuple, H: int = 256, W: int = 256) -> Dict:
    layers = {
        "parks": create_grid(H, W),
        "water": create_grid(H, W),
        "residential": create_grid(H, W),
        "commercial": create_grid(H, W),
        "corridors": create_grid(H, W),
        "barriers": create_grid(H, W),
        "noise_src": create_grid(H, W),
        "pop": create_grid(H, W),
    }

    nodes = {el["id"]: el for el in elements if el["type"] == "node"}

    def draw_node(lat: float, lon: float, grid: np.ndarray, val: float = 1.0):
        r, c = latlon_to_rc(lat, lon, bbox, H, W)
        grid[r, c] += val

    def draw_way(node_ids: List[int], grid: np.ndarray, val: float = 1.0):
        for i in range(1, len(node_ids)):
            a = nodes.get(node_ids[i-1])
            b = nodes.get(node_ids[i])
            if not a or not b:
                continue
            steps = 12
            for t in range(steps + 1):
                lat = a["lat"] + (b["lat"] - a["lat"]) * t / steps
                lon = a["lon"] + (b["lon"] - a["lon"]) * t / steps
                draw_node(lat, lon, grid, val)

    for el in elements:
        tags = el.get("tags", {})
        if el["type"] == "way":
            if tags.get("leisure") == "park" or tags.get("landuse") in ("grass", "forest"):
                draw_way(el["nodes"], layers["parks"], 1)
            if tags.get("natural") == "water" or tags.get("water") or tags.get("waterway"):
                draw_way(el["nodes"], layers["water"], 1)
            if tags.get("landuse") == "residential":
                draw_way(el["nodes"], layers["residential"], 1)
            if tags.get("landuse") == "commercial":
                draw_way(el["nodes"], layers["commercial"], 1)
            if "highway" in tags:
                highway = tags["highway"]
                if any(k in highway for k in ["footway", "path", "cycleway", "residential", "living_street"]):
                    draw_way(el["nodes"], layers["corridors"], 1)
                if any(k in highway for k in ["trunk", "primary", "motorway"]):
                    draw_way(el["nodes"], layers["barriers"], 2)
                    draw_way(el["nodes"], layers["noise_src"], 1.5)
            if "railway" in tags:
                draw_way(el["nodes"], layers["barriers"], 2)
                draw_way(el["nodes"], layers["noise_src"], 1.5)

    layers["noise_src"] = normalize01(blur(blur(layers["noise_src"], 2, 2), 2, 2))
    for key in ["parks", "water", "residential", "commercial", "corridors", "barriers"]:
        layers[key] = normalize01(blur(layers[key], 1, 1))
    layers["pop"] = normalize01(blur(0.6 * layers["residential"] + 0.8 * layers["commercial"], 1, 1))
    return layers

# ============== Model Functions ==============
def alphas_from_profile(dog: Dict, ctx: Dict) -> Dict:
    a_parks = -0.7 * (0.6 + 0.4 * dog["energy"]) * (1.0 - 0.5 * ctx["precip"])
    a_water = -0.5 * (0.4 + 0.6 * dog["water_aff"]) * (1.0 - 0.6 * ctx["precip"])
    a_res   = -0.6 * (0.4 + 0.6 * dog["human_dep"] + 0.3 * dog["sociality"])
    a_com   =  0.5 * (0.5 + 0.5 * dog["noise"]) * (0.7 + 0.3 * ctx["event"])
    a_bar   =  0.9
    a_evt   =  0.7 * (0.6 + 0.4 * dog["noise"]) * (0.5 + 0.5 * ctx["event"])
    a_noise =  1.4 * (0.5 + 0.5 * dog["noise"])
    return {
        "parks": a_parks,
        "water": a_water,
        "residential": a_res,
        "commercial": a_com,
        "barriers": a_bar,
        "event": a_evt,
        "noise": a_noise,
    }

def build_potential(layers: Dict, alphas: Dict) -> np.ndarray:
    U = (alphas["parks"] * layers["parks"] +
         alphas["water"] * layers["water"] +
         alphas["residential"] * layers["residential"] +
         alphas["commercial"] * layers["commercial"] +
         alphas["barriers"] * layers["barriers"] +
         alphas["event"] * layers["pop"] +
         alphas["noise"] * layers["noise_src"])
    mean = float(U.mean())
    std = float(U.std()) + 1e-6
    return (U - mean) / std

def simulate(U: np.ndarray, layers: Dict, dog: Dict, ctx: Dict,
             start_rc: Tuple[int, int], N: int = 3200, steps: int = 230) -> np.ndarray:
    H, W = U.shape
    gy = np.zeros_like(U)
    gx = np.zeros_like(U)
    gy[1:-1, :] = (U[2:, :] - U[:-2, :]) / 2.0
    gx[:, 1:-1] = (U[:, 2:] - U[:, :-2]) / 2.0

    barriers = layers["barriers"]
    corridors = layers["corridors"]

    heat = np.zeros((H, W), dtype=np.float32)
    ys = np.full(N, start_rc[0], dtype=np.float32)
    xs = np.full(N, start_rc[1], dtype=np.float32)
    alive = np.ones(N, dtype=np.uint8)

    base_speed = max(0.3, 1.5 * (0.8 + 0.4 * dog["energy"]) * (0.9 - 0.3 * ctx["precip"]))
    rng = np.random.default_rng()

    for t in range(steps):
        r = np.clip(ys.astype(int), 0, H-1)
        c = np.clip(xs.astype(int), 0, W-1)

        gys = gy[r, c]
        gxs = gx[r, c]
        norm = np.sqrt(gys*gys + gxs*gxs) + 1e-6
        diry = -gys / norm
        dirx = -gxs / norm

        corr = corridors[r, c]
        speed = base_speed * (0.8 + 0.6 * corr)

        diry += (rng.random(N) - 0.5) * 0.6
        dirx += (rng.random(N) - 0.5) * 0.6

        ys += diry * speed
        xs += dirx * speed

        ys = np.clip(ys, 0, H-1)
        xs = np.clip(xs, 0, W-1)

        idx = barriers[r, c] > 0.6
        ys[idx] -= diry[idx] * 0.6
        xs[idx] -= dirx[idx] * 0.6

        if t % 10 == 0:
            die = rng.random(N) < 0.005
            alive[die] = 0

    alive_idx = np.where(alive == 1)[0]
    for i in alive_idx:
        r = int(ys[i]); c = int(xs[i])
        heat[r, c] += 1.0

    tot = float(heat.sum())
    if tot > 0:
        heat /= tot
    return blur(heat, 1, 1)

# ============== Visualization ==============
def heat_rgba_array(heat: np.ndarray, q: float = 0.80, gamma: float = 0.65) -> np.ndarray:
    H, W = heat.shape
    thr = float(np.quantile(heat, q))
    img = np.zeros((H, W, 4), dtype=np.uint8)
    v = heat.copy()
    v[v < thr] = thr
    v = (v - thr) / (1 - thr + 1e-9)
    v = np.clip(v, 0, 1) ** gamma
    img[:, :, 0] = (255 * v).astype(np.uint8)
    img[:, :, 1] = (90 * (1 - v)).astype(np.uint8)
    img[:, :, 2] = (160 * (1 - v)).astype(np.uint8)
    img[:, :, 3] = (255 * v).astype(np.uint8)
    return img

def rgba_to_data_url(arr: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(arr, 'RGBA').save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

def top_hotspots(heat: np.ndarray, bbox: Tuple, k: int = 5) -> List[Dict]:
    H, W = heat.shape
    flat = heat.flatten()
    idxs = np.argpartition(-flat, range(min(k, flat.size)))[:k]
    idxs = idxs[np.argsort(-flat[idxs])]
    pts = []
    for idx in idxs:
        r = int(idx // W)
        c = int(idx % W)
        lat, lon = rc_to_latlon(r, c, bbox, H, W)
        pts.append({"lat": lat, "lon": lon, "score": float(heat[r, c])})
    return pts

def compute_route(lat0: float, lon0: float, hotspots: List[Dict], k: int = 5) -> Dict:
    hs = hotspots[:k]
    visited = [False] * len(hs)
    order = []
    lat, lon = lat0, lon0
    for _ in range(len(hs)):
        best_d, best_i = 1e9, -1
        for i, h in enumerate(hs):
            if visited[i]:
                continue
            d = haversine_km(lat, lon, h["lat"], h["lon"])
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            order.append(best_i)
            visited[best_i] = True
            lat, lon = hs[best_i]["lat"], hs[best_i]["lon"]
    points = [{"lat": lat0, "lon": lon0}]
    for i in order:
        points.append({"lat": hs[i]["lat"], "lon": hs[i]["lon"]})
    total_km = sum(
        haversine_km(points[i]["lat"], points[i]["lon"], points[i+1]["lat"], points[i+1]["lon"])
        for i in range(len(points)-1)
    )
    return {"order": order, "points": points, "distance_km": total_km}

# ============== Reasoning Text ==============
def generate_reasoning(dog: Dict, ctx: Dict, attributions: List[Dict], hotspots: List[Dict]) -> List[str]:
    reasons = []
    if dog['noise'] > 0.7:
        reasons.append(f"🔊 소음에 매우 민감(민감도 {dog['noise']:.1%}) → 큰 도로나 번화 상업지역 회피 경향.")
    elif dog['noise'] < 0.3:
        reasons.append(f"🔊 소음 둔감(민감도 {dog['noise']:.1%}) → 도심에서도 활동 가능.")
    if dog['human_dep'] > 0.7:
        reasons.append(f"👥 사람 의존 높음(의존도 {dog['human_dep']:.1%}) → 주거지역 근처 선호.")
    elif dog['human_dep'] < 0.3:
        reasons.append(f"👤 독립 성향(의존도 {dog['human_dep']:.1%}) → 인적 드문 곳 선호.")
    if dog['energy'] > 0.7:
        reasons.append(f"⚡ 높은 에너지(활동성 {dog['energy']:.1%}) → 이동 반경 넓을 가능성.")
    elif dog['energy'] < 0.3:
        reasons.append(f"😴 낮은 에너지(활동성 {dog['energy']:.1%}) → 출발지 인근 머무를 가능성.")
    if ctx['precip'] > 0.5:
        reasons.append("🌧️ 강수 영향 → 비 피할 수 있는 구조물 선호(다리 밑/처마/주차장 등).")
    if ctx['event'] > 0.6:
        reasons.append("🎭 이벤트/축제 영향 → 발견 확률↑ 또는 소음 회피로 원거리 이동 가능성.")
    if attributions:
        top_factor = attributions[0]
        if top_factor['contribution'] < 0:
            reasons.append(f"📍 '{top_factor['factor']}' 요인이 강한 끌림으로 작용.")
        else:
            reasons.append(f"⚠️ '{top_factor['factor']}' 요인을 회피하는 경향.")
    if hotspots:
        reasons.append(f"🎯 최상위 후보지 점수 {hotspots[0]['score']:.3f} (순위 1).")
    return reasons

# ============== API Endpoints ==============
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        data = request.get_json(silent=True) if request.method == 'POST' else dict(request.args)
        data = data or {}

        lat = float(data.get('lat', 37.5665))
        lon = float(data.get('lon', 126.9780))

        dog = {
            'energy': float(data.get('energy', 0.7)),
            'sociality': float(data.get('sociality', 0.7)),
            'noise': float(data.get('noise', 0.85)),
            'water_aff': float(data.get('water', 0.4)),
            'human_dep': float(data.get('human', 0.8)),
            'size': float(data.get('size', 0.2)),
            'cute': float(data.get('cute', 0.9)),
            'clean': float(data.get('clean', 0.8)),
            'collar': float(data.get('collar', 1.0)),
        }
        ctx = {
            'event': float(data.get('event', 0.4)),
            'precip': float(data.get('precip', 0.3)),
        }
        box_km = float(data.get('box_km', 2.5))

        cache_key = get_cache_key(lat, lon, {'dog': dog, 'ctx': ctx, 'box_km': box_km})
        if cache_key in cache_data:
            cached_result, cached_time = cache_data[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=30):
                return jsonify(cached_result)

        bbox = small_bbox(lat, lon, km=box_km)
        osm_data = fetch_osm(bbox)
        if not osm_data.get('elements'):
            bbox = small_bbox(lat, lon, km=max(1.5, box_km * 0.7))
            osm_data = fetch_osm(bbox)
        if not osm_data.get('elements'):
            return jsonify({'error': 'no_data', 'message': 'OSM 데이터를 가져올 수 없습니다'}), 404

        H = W = 256
        layers = rasterize(osm_data['elements'], bbox, H, W)
        alphas = alphas_from_profile(dog, ctx)
        U = build_potential(layers, alphas)

        sr, sc = latlon_to_rc(lat, lon, bbox, H, W)
        heat = simulate(U, layers, dog, ctx, (sr, sc))
        if float(heat.std()) < 1e-9:
            U0 = normalize01(U)
            heat = blur(1 - U0, 1, 1)

        rgba = heat_rgba_array(heat)
        image_url = rgba_to_data_url(rgba)
        hotspots = top_hotspots(heat, bbox, k=5)
        route = compute_route(lat, lon, hotspots, k=5)

        thr = float(np.quantile(heat, 0.9))
        mask = heat >= thr
        attributions = []
        factor_names = {
            'parks': '공원/녹지',
            'water': '물/하천',
            'residential': '주거지역',
            'commercial': '상업지역',
            'barriers': '도로/철도',
            'noise_src': '소음원',
        }
        for key, display in factor_names.items():
            level = float(layers[key][mask].mean()) if mask.any() else 0.0
            weight = alphas.get(key if key != 'noise_src' else 'noise', 0.0)
            contrib = weight * level
            attributions.append({'factor': display, 'level': level, 'weight': weight, 'contribution': contrib})
        attributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        reasoning = generate_reasoning(dog, ctx, attributions, hotspots)
        pickup_score = (0.8 * dog['cute'] + 0.5 * dog['clean'] -
                        0.6 * dog['collar'] - 0.7 * dog['size'] + 0.35 * ctx['event'])

        result = {
            'bounds': {'s': bbox[0], 'w': bbox[1], 'n': bbox[2], 'e': bbox[3]},
            'image_data_url': image_url,
            'hotspots': hotspots,
            'route': route,
            'attribution': attributions[:6],
            'reasoning': reasoning,
            'pickup_risk': {
                'score': pickup_score,
                'level': 'high' if pickup_score > 0.5 else 'medium' if pickup_score > 0 else 'low'
            },
            'origin': {'lat': lat, 'lon': lon},
            'confidence': 0.75
        }

        cache_data[cache_key] = (result, datetime.now())
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': 'server_error', 'message': str(e)}), 500

# ============== HTML Template ==============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8">
    <title>유기견 위치 예측 시스템</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        #map { height: 100vh; width: 100%; }
        .control-panel { position: fixed; top: 20px; left: 20px; width: 360px; background: rgba(255,255,255,0.95); border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 20px; z-index: 1000; max-height: calc(100vh - 40px); overflow-y: auto; }
        .title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #333; }
        .slider-group { margin-bottom: 12px; }
        .slider-label { display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 13px; color: #555; }
        input[type="range"] { width: 100%; height: 6px; border-radius: 3px; background: #ddd; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: #4CAF50; cursor: pointer; }
        .btn { width: 100%; padding: 12px; background: #4CAF50; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 15px; }
        .btn:hover { background: #45a049; }
        .info-panel { position: fixed; top: 20px; right: 20px; width: 360px; background: rgba(255,255,255,0.95); border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 20px; z-index: 1000; display: none; max-height: 400px; overflow-y: auto; }
        .reasoning-panel { position: fixed; bottom: 20px; right: 20px; width: 400px; background: rgba(255,255,255,0.98); border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); padding: 20px; z-index: 1000; display: none; max-height: 300px; overflow-y: auto; }
        .reasoning-title { font-size: 16px; font-weight: bold; margin-bottom: 12px; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }
        .reasoning-item { padding: 8px 12px; margin: 8px 0; background: #f8f9fa; border-left: 4px solid #4CAF50; border-radius: 4px; font-size: 14px; line-height: 1.5; color: #2c3e50; }
        .hotspot-item { padding: 10px; margin: 8px 0; background: #f5f5f5; border-radius: 8px; cursor: pointer; transition: all 0.3s; border: 2px solid transparent; }
        .hotspot-item:hover { background: #e8f5e9; border-color: #4CAF50; transform: translateX(5px); }
        .hotspot-number { display: inline-block; width: 28px; height: 28px; background: #4CAF50; color: white; border-radius: 50%; text-align: center; line-height: 28px; font-weight: bold; margin-right: 10px; }
        .hotspot-score { float: right; background: #fff3cd; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; color: #856404; }
        .status { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); color: white; padding: 10px 20px; border-radius: 20px; z-index: 1000; display: none; }
        .risk-indicator { padding: 8px 12px; margin-top: 10px; border-radius: 6px; font-size: 14px; font-weight: bold; text-align: center; }
        .risk-high { background: #ffebee; color: #c62828; border: 2px solid #ef5350; }
        .risk-medium { background: #fff3e0; color: #e65100; border: 2px solid #ff9800; }
        .risk-low { background: #e8f5e9; color: #1b5e20; border: 2px solid #4CAF50; }
        .custom-marker { background: none; border: none; }
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="control-panel">
        <div class="title">🐕 유기견 위치 예측</div>
        <div class="slider-group"><div class="slider-label"><span>소음 민감도</span><span id="noise-val">0.85</span></div><input type="range" id="noise" min="0" max="1" step="0.05" value="0.85"></div>
        <div class="slider-group"><div class="slider-label"><span>사람 의존도</span><span id="human-val">0.8</span></div><input type="range" id="human" min="0" max="1" step="0.05" value="0.8"></div>
        <div class="slider-group"><div class="slider-label"><span>에너지</span><span id="energy-val">0.7</span></div><input type="range" id="energy" min="0" max="1" step="0.05" value="0.7"></div>
        <div class="slider-group"><div class="slider-label"><span>이벤트</span><span id="event-val">0.4</span></div><input type="range" id="event" min="0" max="1" step="0.05" value="0.4"></div>
        <div class="slider-group"><div class="slider-label"><span>강수량</span><span id="precip-val">0.3</span></div><input type="range" id="precip" min="0" max="1" step="0.05" value="0.3"></div>
        <div class="slider-group"><div class="slider-label"><span>검색 반경 (km)</span><span id="box-val">2.5</span></div><input type="range" id="box" min="1" max="5" step="0.5" value="2.5"></div>
        <button class="btn" onclick="runPrediction()">예측 시작</button>
        <div id="risk-indicator"></div>
    </div>
    <div class="info-panel" id="info-panel">
        <div class="title">📍 예측 위치 (상위 5개)</div>
        <div id="hotspot-list"></div>
    </div>
    <div class="reasoning-panel" id="reasoning-panel">
        <div class="reasoning-title">🧠 AI 분석 근거</div>
        <div id="reasoning-list"></div>
    </div>
    <div class="status" id="status">예측 중...</div>

    <script>
        const map = L.map('map').setView([37.5665, 126.9780], 13);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© OpenStreetMap'}).addTo(map);

        let heatmapOverlay = null;
        let hotspotMarkers = L.layerGroup().addTo(map);
        let routeLayer = L.layerGroup().addTo(map);

        document.querySelectorAll('input[type="range"]').forEach(slider => {
            slider.addEventListener('input', (e) => {
                document.getElementById(e.target.id + '-val').textContent = e.target.value;
            });
        });

        function showStatus(msg) {
            const status = document.getElementById('status');
            status.textContent = msg || '예측 중...';
            status.style.display = 'block';
        }
        function hideStatus() { document.getElementById('status').style.display = 'none'; }

        async function runPrediction() {
            const center = map.getCenter();
            await predictAt(center.lat, center.lng);
        }

        async function predictAt(lat, lng) {
            showStatus('AI가 분석 중입니다...');
            const params = new URLSearchParams({
                lat: lat, lon: lng,
                noise: document.getElementById('noise').value,
                human: document.getElementById('human').value,
                energy: document.getElementById('energy').value,
                event: document.getElementById('event').value,
                precip: document.getElementById('precip').value,
                box_km: document.getElementById('box').value
            });
            try {
                const response = await fetch('/predict?' + params);
                const result = await response.json();
                if (result.error) {
                    alert('오류: ' + (result.message || result.error));
                    hideStatus();
                    return;
                }
                displayResults(result);
                hideStatus();
            } catch (error) {
                alert('오류가 발생했습니다: ' + error.message);
                hideStatus();
            }
        }

        function displayResults(result) {
            if (heatmapOverlay) map.removeLayer(heatmapOverlay);
            hotspotMarkers.clearLayers();
            routeLayer.clearLayers();

            if (result.image_data_url && result.bounds) {
                const bounds = [[result.bounds.s, result.bounds.w],[result.bounds.n, result.bounds.e]];
                heatmapOverlay = L.imageOverlay(result.image_data_url, bounds, {opacity: 0.7}).addTo(map);
                map.fitBounds(bounds);
            }

            if (result.hotspots) {
                const hotspotHTML = result.hotspots.slice(0,5).map((spot, idx) => {
                    const icon = L.divIcon({
                        className: 'custom-marker',
                        html: `<div class="hotspot-number">${idx + 1}</div>`,
                        iconSize: [30, 30]
                    });
                    L.marker([spot.lat, spot.lon], {icon: icon})
                        .bindPopup(`
                            <strong>예측 순위 #${idx + 1}</strong><br>
                            확률 점수: ${spot.score.toFixed(3)}<br>
                            좌표: ${spot.lat.toFixed(5)}, ${spot.lon.toFixed(5)}
                        `)
                        .addTo(hotspotMarkers);
                    return `
                        <div class="hotspot-item" onclick="map.setView([${spot.lat}, ${spot.lon}], 16)">
                            <span class="hotspot-number">${idx + 1}</span>
                            <span>위도: ${spot.lat.toFixed(5)}, 경도: ${spot.lon.toFixed(5)}</span>
                            <span class="hotspot-score">${(spot.score * 100).toFixed(1)}%</span>
                        </div>
                    `;
                }).join('');
                document.getElementById('hotspot-list').innerHTML = hotspotHTML;
                document.getElementById('info-panel').style.display = 'block';
            }

            if (result.route && result.route.points) {
                const latlngs = result.route.points.slice(0,6).map(p => [p.lat, p.lon]);
                L.polyline(latlngs, {color: '#2196F3', weight: 3, opacity: 0.7, dashArray: '10,5'}).addTo(routeLayer);
            }

            if (result.reasoning) {
                const reasoningHTML = result.reasoning.map(reason => `<div class="reasoning-item">${reason}</div>`).join('');
                document.getElementById('reasoning-list').innerHTML = reasoningHTML;
                document.getElementById('reasoning-panel').style.display = 'block';
            }

            if (result.pickup_risk) {
                const risk = result.pickup_risk;
                const riskClass = `risk-${risk.level}`;
                const riskText = {'high': '⚠️ 픽업 위험도: 높음','medium': '⚡ 픽업 위험도: 보통','low': '✅ 픽업 위험도: 낮음'};
                document.getElementById('risk-indicator').innerHTML = `
                    <div class="risk-indicator ${riskClass}">
                        ${riskText[risk.level]}<br><small>점수: ${risk.score.toFixed(2)}</small>
                    </div>
                `;
            }
        }

        map.on('click', (e) => { predictAt(e.latlng.lat, e.latlng.lng); });
        setTimeout(runPrediction, 1000);
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return Response(HTML_TEMPLATE, mimetype='text/html')

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════╗
    ║   유기견 위치 예측 시스템              ║
    ║   http://localhost:5000                ║
    ╚════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=True)
