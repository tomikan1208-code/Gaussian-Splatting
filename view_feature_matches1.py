import argparse
from contextlib import closing
from functools import lru_cache
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import struct
import sys
from urllib.parse import urlsplit, parse_qs, unquote
from threading import RLock
import webbrowser

from room_data import RUN_NAME, read_json
from room_geometry import pycolmap


ROOT = Path(__file__).resolve().parent
PAIR_ID_LIMIT = 2147483647
sys.path.insert(0, str(ROOT / "apps/heif-python-packages"))
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


import math
import numpy as np


def _apply(h, xy):
    p = np.column_stack((xy, np.ones(len(xy)))) @ h.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return p[:, :2] / p[:, 2:]


def _similarity(a, b):
    # b -> a, positive scale, no reflection
    x = np.column_stack((b[:, 0], -b[:, 1], np.ones(len(b)), np.zeros(len(b))))
    y = np.column_stack((b[:, 1], b[:, 0], np.zeros(len(b)), np.ones(len(b))))
    m = np.stack((x, y), axis=1).reshape(-1, 4)
    v, _, rank, _ = np.linalg.lstsq(m, a.reshape(-1), rcond=None)
    if rank < 4 or np.hypot(*v[:2]) < 1e-7:
        raise ValueError('degenerate similarity')
    c, s, tx, ty = v
    return np.array([[c, -s, tx], [s, c, ty], [0, 0, 1.]])


def _normalize(x):
    center = x.mean(axis=0)
    scale = np.sqrt(2) / max(np.mean(np.linalg.norm(x-center, axis=1)), 1e-9)
    t = np.array([[scale, 0, -scale*center[0]], [0, scale, -scale*center[1]], [0, 0, 1.]])
    return _apply(t, x), t


def _homography(a, b):
    a, ta = _normalize(a); b, tb = _normalize(b)
    x, y = b.T; u, v = a.T; o = np.ones(len(a)); z = np.zeros(len(a))
    m = np.stack((np.column_stack((x,y,o,z,z,z,-u*x,-u*y,-u)),
                  np.column_stack((z,z,z,x,y,o,-v*x,-v*y,-v))),axis=1).reshape(-1,9)
    _, s, vt = np.linalg.svd(m, full_matrices=len(m)<9)
    if len(s)<8 or s[7] < 1e-9:
        raise ValueError('degenerate homography')
    h = np.linalg.solve(ta, vt[-1].reshape(3,3) @ tb)
    if abs(h[2,2]) < 1e-8:
        raise ValueError('invalid projective center')
    return h / h[2,2]


def _valid(h, corners):
    if not np.isfinite(h).all(): return False
    w = corners @ h[2,:2] + h[2,2]
    if np.min(w) < .12 or np.max(w) > 8: return False
    p = _apply(h, corners)
    if not np.isfinite(p).all() or np.max(np.abs(p)) > 8: return False
    try: return abs(np.linalg.det(h)) > 1e-5 and np.linalg.cond(h) < 1000
    except np.linalg.LinAlgError: return False


def _ransac(a,b,fit,n,threshold,corners,iterations):
    if len(a)<n: return None, np.zeros(len(a),dtype=bool)
    rng = np.random.default_rng(42)
    best = None; mask = np.zeros(len(a),dtype=bool); score = (-1, -np.inf)
    # Limit hypothesis scoring for very dense pairs; refine against all points later.
    subset = rng.choice(len(a), min(len(a), 2500), replace=False)
    aa,bb = a[subset],b[subset]
    for i in range(iterations if len(a)>n else 1):
        try:
            ix = rng.choice(len(aa),n,replace=False)
            h = fit(aa[ix],bb[ix])
            if not _valid(h,corners): continue
            e = np.linalg.norm(_apply(h,bb)-aa,axis=1)
            m = e < threshold
            sc = (int(m.sum()), -float(np.minimum(e,threshold).sum()))
            if sc > score:
                best,score = h,sc
                if sc[0] == len(aa): break
        except (ValueError,np.linalg.LinAlgError,FloatingPointError): continue
    if best is None: return None,mask
    for _ in range(3):
        mask = np.linalg.norm(_apply(best,b)-a,axis=1)<threshold
        if mask.sum()<n: return None,mask
        try:
            h = fit(a[mask],b[mask])
            if _valid(h,corners): best=h
        except (ValueError,np.linalg.LinAlgError): break
    mask = np.linalg.norm(_apply(best,b)-a,axis=1)<threshold
    return best,mask


def estimate(points, left_size, right_size):
    """Coordinates centered on each image, long edge = 1; H maps right -> left."""
    if len(points)<2: return None
    p = np.asarray(points,dtype=float).reshape(-1,4)
    a = (p[:,:2]-np.asarray(left_size)/2)/max(left_size)
    b = (p[:,2:]-np.asarray(right_size)/2)/max(right_size)
    w,h = np.asarray(right_size)/max(right_size)/2
    corners = np.array([[-w,-h],[w,-h],[w,h],[-w,h]])
    sim,sm = _ransac(a,b,_similarity,2,.035,corners,400)
    if sim is None: return None
    hom,hm = _ransac(a,b,_homography,4,.012,corners,750) if len(a)>=4 else (None,np.zeros(len(a),bool))
    # Avoid implausible four-point fits on generally nonplanar scenes.
    use_h = hom is not None and hm.sum()>=max(4 if len(a)<8 else 8, math.ceil(.12*len(a)))
    matrix,model = (hom,'homography') if use_h else (sim,'similarity')
    residual = np.linalg.norm(_apply(matrix,b)-a,axis=1)
    mask = residual < .012
    return {'matrix':matrix.tolist(),'model':model,
            'angle':math.degrees(math.atan2(sim[1,0],sim[0,0])),
            'rotation_inliers':int(sm.sum()),'inliers':int(mask.sum()),'total':len(a),
            'median_px':float(np.median(residual)*max(left_size))}



APP_HTML = r'''<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>写真の特徴点照合</title>
  <style>
    :root {
      color-scheme: light;
      --paper: #edf2ef;
      --ink: #142b2e;
      --muted: #4b6465;
      --rule: #bbc9c5;
      --stage: #183237;
      --accent: #046254;
      font-family: "Yu Gothic UI", Meiryo, system-ui, sans-serif;
      color: var(--ink);
      background: var(--paper);
    }
    * { box-sizing: border-box; }
    body { margin: 0; }
    main { width: min(1440px, calc(100% - 48px)); margin: 0 auto; padding: 24px 0 30px; }
    h1 { margin: 0 0 4px; font-size: clamp(1.55rem, 3vw, 2.35rem); letter-spacing: -.035em; line-height: 1.2; }
    p { margin: 0; }
    [hidden] { display: none !important; }
    #summary { color: var(--muted); font-size: .9rem; margin-top: 7px; }
    .examples { display: flex; flex-wrap: wrap; gap: 6px; margin: 15px 0; }
    button, select {
      font: inherit;
      color: inherit;
      border: 1px solid var(--rule);
      border-radius: 4px;
      background: #fff;
    }
    button { cursor: pointer; padding: 5px 10px; }
    button:hover, button[aria-pressed="true"] { background: #dceee7; border-color: var(--accent); }
    :focus-visible { outline: 3px solid #167b66; outline-offset: 2px; }
    .selectors { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .selector { min-width: 0; padding: 9px 0 10px; border-top: 1px solid var(--rule); }
    .selector-label { margin-bottom: 5px; font-size: .85rem; font-weight: 700; }
    .photo-choice {
      display: flex; align-items: center; gap: 9px; width: 100%; min-width: 0;
      padding: 5px 8px; text-align: left;
    }
    .photo-thumbnail { display: block; flex: none; object-fit: contain; background: #233b3e; }
    .photo-thumbnail.unavailable { opacity: .45; }
    .photo-choice .photo-thumbnail { width: 62px; height: 49px; border-radius: 2px; }
    .photo-label { min-width: 0; display: grid; gap: 2px; }
    .photo-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .photo-detail { color: var(--muted); font-size: .77rem; }
    .metadata { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 9px; margin-top: 8px; color: var(--muted); font-size: .78rem; }
    .metadata button { margin-left: auto; white-space: nowrap; }
    .status-bar {
      display: flex; align-items: center; flex-wrap: wrap; gap: 8px 16px;
      padding: 9px 12px; background: #d9e6e0;
    }
    #pair-status { font-size: 1.04rem; font-variant-numeric: tabular-nums; }
    #pair-support { color: #154d42; font-size: .84rem; }
    #display-count { color: var(--muted); font-size: .84rem; margin-right: auto; }
    .controls { display: flex; align-items: center; gap: 11px; font-size: .84rem; }
    .controls select { padding: 3px 5px; }
    .controls input { accent-color: var(--accent); }
    .stage {
      height: clamp(320px, 59vh, 680px);
      background: var(--stage);
      min-width: 0;
      position: relative;
      overflow: hidden;
    }
    .stage canvas { display: block; width: 100%; height: 100%; touch-action: none; position: relative; z-index: 3; }
    .compare-plane { position:absolute; top:0; left:0; transform-origin:0 0; pointer-events:none; display:none; outline:1px solid #8fdfc5; }
    #plane-right { outline-color:#ffc17a; }
    .compare-tools { display:flex; flex-wrap:wrap; align-items:center; gap:6px 12px; padding:8px 12px; border-bottom:1px solid var(--rule); background:#e7eeea; font-size:.84rem; }
    .compare-tools label { display:inline-flex; align-items:center; gap:4px; }
    .compare-tools input[type=range] { width:95px; accent-color:var(--accent); }
    .compare-tools output { min-width:30px; font-variant-numeric:tabular-nums; }
    .controls { flex-wrap:wrap; }
    .stage canvas:focus-visible { outline-color: #b6f4a8; outline-offset: -3px; }
    #point-note {
      padding: 10px 12px; color: var(--muted); font-size: .85rem; min-height: 40px;
      border-bottom: 1px solid var(--rule); font-variant-numeric: tabular-nums;
    }
    #error { padding: 12px 0; color: #85451d; }
    .view-switch { display: flex; gap: 6px; padding: 12px 0; }
    .graph-toolbar, .graph-types, .graph-zoom, .relation-tabs {
      display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px;
    }
    .graph-toolbar { padding: 9px 10px; background: #d9e6e0; font-size: .83rem; }
    .graph-toolbar select { padding: 3px 6px; }
    .graph-types { color: var(--muted); margin-right: auto; }
    .graph-types span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
    .graph-types .registered::before { background: #087565; }
    .graph-types .unregistered::before { background: #b36719; }
    .graph-types .isolated::before { border: 1.5px solid #875426; }
    .graph-zoom button { padding: 3px 9px; }
    .graph-layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(270px, 25%); }
    .graph-stage { position: relative; min-width: 0; height: clamp(430px, 65vh, 810px); background: var(--stage); }
    .graph-stage canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
    .graph-stage canvas:focus-visible { outline-color: #b6f4a8; outline-offset: -3px; }
    #graph-tip { padding: 7px 10px; min-height: 32px; background: #13292d; color: #e5f0eb; font-size: .75rem; }
    .graph-sidebar { min-width: 0; border: 1px solid var(--rule); padding: 11px; background: #f7faf7; }
    .graph-sidebar .photo-choice { margin: 5px 0 8px; font-size: .85rem; }
    .graph-pick { margin-top: 8px; display: flex; flex-wrap: wrap; align-items: center; gap: 4px; }
    .graph-pick button { padding: 4px 7px; font-size: .75rem; }
    #graph-pending { margin-top: 5px; color: var(--accent); font-size: .78rem; }
    #node-status { color: var(--muted); font-size: .79rem; }
    .graph-path { border-top: 1px solid var(--rule); padding: 10px 0; margin-top: 9px; }
    .graph-path h2 { font-size: .87rem; margin: 0 0 6px; }
    .graph-path p { font-size: .77rem; margin: 5px 0; color: var(--muted); }
    .graph-path button { margin: 2px 3px 2px 0; font-size: .75rem; }
    .relation-tabs { gap: 4px; }
    .relation-tabs button { padding: 4px 6px; font-size: .73rem; }
    .relation-list { margin-top: 5px; max-height: 240px; overflow-y: auto; }
    .relation-list button { margin: 3px 0; font-size: .75rem; }
    .relation-list .photo-thumbnail { width: 46px; height: 40px; }
    .relation-list p { font-size: .77rem; color: var(--muted); padding: 5px 0; }
    .photo-dialog {
      width: min(980px, calc(100% - 24px)); max-height: min(800px, 87vh);
      padding: 0; color: var(--ink); background: var(--paper); border: 1px solid var(--rule);
      box-shadow: 0 14px 65px #13292d55;
    }
    .photo-dialog::backdrop { background: #13292da6; }
    .gallery-head {
      position: sticky; top: 0; z-index: 1; display: flex; flex-wrap: wrap; align-items: center;
      gap: 7px 12px; padding: 12px 16px; background: var(--paper); border-bottom: 1px solid var(--rule);
    }
    .gallery-head h2 { margin: 0 auto 0 0; font-size: 1.16rem; }
    .gallery-head input {
      max-width: 100%; font: inherit; color: inherit; padding: 5px 8px;
      border: 1px solid var(--rule); border-radius: 4px;
    }
    .gallery {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(133px, 1fr)); gap: 9px; padding: 12px 16px;
    }
    .gallery-card { min-width: 0; text-align: left; padding: 4px; }
    .gallery-card .photo-thumbnail { width: 100%; aspect-ratio: 4 / 3; }
    .gallery-card .photo-label { padding: 5px 3px; font-size: .83rem; }
    @media (max-width: 640px) {
      main { width: calc(100% - 24px); padding-top: 16px; }
      .selectors { gap: 8px; }
      .photo-choice { font-size: .8rem; gap: 5px; }
      .photo-choice .photo-thumbnail { width: 43px; height: 38px; }
      .metadata { font-size: .72rem; gap: 4px 6px; }
      .metadata button { margin-left: 0; }
      .stage { height: clamp(230px, 65vw, 365px); }
      .graph-layout { grid-template-columns: 1fr; }
      .graph-stage { height: clamp(340px, 100vw, 530px); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>写真の特徴点照合</h1>
      <p id="summary">照合データを読み込んでいます…</p>
      <div class="compare-tools" style="margin-top:8px;background:transparent;padding:0">
        <label>run <select id="run-select" aria-label="実行フォルダ"></select></label>
        <label title="撮影時のスマホの姿勢を使い、重力方向が下になるよう補正">地面基準の自動回転 <select id="rotation-global"><option value="90">90°刻み</option><option value="free">無制限</option><option value="off">なし</option></select></label>
        <span id="orientation-summary"></span>
      </div>
    </header>
    <p id="error" role="alert" hidden></p>
    <nav class="view-switch" id="view-switch" aria-label="表示" hidden>
      <button type="button" id="show-graph" aria-pressed="true">全体グラフ</button>
      <button type="button" id="show-comparison" aria-pressed="false">写真の比較</button>
    </nav>
    <section id="graph-view" aria-label="照合グラフ" hidden>
      <div class="graph-toolbar">
        <label>表示 <select id="graph-display">
          <option value="photos">写真</option><option value="numbers">番号</option>
        </select></label>
        <label>一致点数の下限 <select id="graph-threshold">
          <option value="1">すべて</option><option value="30">30点以上</option>
          <option value="75">75点以上</option><option value="150">150点以上</option>
          <option value="300">300点以上</option>
        </select></label>
        <span id="graph-count"></span>
        <div class="graph-types"><span class="registered">登録済み</span><span class="unregistered">未登録</span>
          <span class="isolated">つながりなし</span></div>
        <div class="graph-zoom" aria-label="グラフの拡大縮小">
          <button type="button" id="graph-out" aria-label="縮小">−</button>
          <button type="button" id="graph-in" aria-label="拡大">＋</button>
          <button type="button" id="graph-reset">全体に戻す</button>
        </div>
      </div>
      <div class="graph-toolbar" style="border-top:1px solid var(--rule)">
        <label><input type="checkbox" id="graph-live"> 動的レイアウト</label>
        <label>リンクの強さ <input id="force-link" type="range" min="0" max="200" value="60"><output id="force-link-value">60</output></label>
        <label>リンク距離 <input id="force-distance" type="range" min="35" max="200" value="65"><output id="force-distance-value">65</output></label>
        <label>反発力 <input id="force-repel" type="range" min="0" max="200" value="80"><output id="force-repel-value">80</output></label>
        <label>中心への引力 <input id="force-center" type="range" min="0" max="200" value="35"><output id="force-center-value">35</output></label>
        <button type="button" id="graph-layout-reset">配置を初期化</button>
        <span id="force-status">固定表示</span>
      </div>
      <div class="graph-layout">
        <div>
          <div class="graph-stage"><canvas id="graph-canvas" aria-label="撮影写真のつながり"></canvas></div>
          <p id="graph-tip">左クリックで左、右クリックで右。2枚そろうと比較できます。</p>
        </div>
        <aside class="graph-sidebar" aria-label="選んだ写真の照合先">
          <p class="selector-label">選んだ写真</p>
          <button class="photo-choice" type="button" id="graph-photo" aria-label="グラフの写真を選ぶ" aria-haspopup="dialog"></button>
          <p id="node-status"></p>
          <div class="graph-pick">
            <button type="button" id="graph-pick-left">左に選ぶ</button>
            <button type="button" id="graph-pick-right">右に選ぶ</button>
            <button type="button" id="graph-pick-clear" hidden>選択解除</button>
          </div>
          <p id="graph-pending" role="status" hidden></p>
          <div class="graph-path" id="graph-path"></div>
          <div class="relation-tabs" aria-label="照合の結果">
            <button type="button" data-relation="match">一致</button>
            <button type="button" data-relation="zero">照合0点</button>
            <button type="button" data-relation="unseen">未照合</button>
          </div>
          <div class="relation-list" id="graph-relations"></div>
        </aside>
      </div>
    </section>
    <section id="comparison" aria-label="写真の比較" hidden>
      <nav class="examples" id="examples" aria-label="比較例"></nav>
      <div class="selectors">
        <div class="selector">
          <p class="selector-label">左の写真</p>
          <button class="photo-choice" type="button" id="left" aria-label="左の写真を選ぶ" aria-haspopup="dialog"></button>
          <div class="metadata"><span id="left-features"></span>
            <button type="button" id="left-rotate" aria-label="左の写真を90度回転">90°回転</button>
          </div>
        </div>
        <div class="selector">
          <p class="selector-label">右の写真</p>
          <button class="photo-choice" type="button" id="right" aria-label="右の写真を選ぶ" aria-haspopup="dialog"></button>
          <div class="metadata"><span id="right-features"></span>
            <button type="button" id="right-rotate" aria-label="右の写真を90度回転">90°回転</button>
          </div>
        </div>
      </div>
      <div class="status-bar">
        <strong id="pair-status" role="status" title="2枚間で幾何検証を通った対応点"></strong>
        <span id="pair-support" title="保存済みのカメラ推定モデルにひもづく対応点"></span>
        <span id="display-count"></span>
        <div class="controls">
          <label>点数 <select id="limit">
            <option value="40">40点</option><option value="120">120点</option>
            <option value="300">300点</option><option value="all">全点</option>
          </select></label>
          <label><input type="checkbox" id="lines" checked> 線を表示</label>
        </div>
      </div>
      <div class="compare-tools">
        <button type="button" id="mode-side" aria-pressed="true">横並び</button>
        <button type="button" id="mode-three" aria-pressed="false">3D重ね合わせ</button>
        <label><input type="checkbox" id="dots" checked> 特徴点を表示</label>
        <button type="button" id="compare-reset">リセット</button><span id="compare-zoom">100%</span>
        <span id="fit-info" role="status"></span>
      </div>
      <div class="compare-tools" id="three-controls" hidden>
        <button type="button" id="auto-align" disabled>自動位置合わせ</button>
        <button type="button" id="front-view">正面視点</button>
        <label>左不透明度 <input id="opacityA" type="range" min="0" max="100" value="60"><output id="opacityAv">60%</output></label>
        <label>右不透明度 <input id="opacityB" type="range" min="0" max="100" value="60"><output id="opacityBv">60%</output></label>
        <label>X傾き <input id="rx" type="range" min="-180" max="180" value="0"><output id="rxv">0°</output></label>
        <label>Y傾き <input id="ry" type="range" min="-180" max="180" value="0"><output id="ryv">0°</output></label>
        <label>Z回転 <input id="rz" type="range" min="-180" max="180" value="0"><output id="rzv">0°</output></label>
        <label>X移動 <input id="tx" type="range" min="-150" max="150" value="0"><output id="txv">0</output></label>
        <label>Y移動 <input id="ty" type="range" min="-150" max="150" value="0"><output id="tyv">0</output></label>
        <label>奥行 <input id="tz" type="range" min="-100" max="150" value="0"><output id="tzv">0</output></label>
        <label>倍率 <input id="sc" type="range" min="25" max="300" value="100"><output id="scv">100%</output></label>
      </div>
      <div class="stage">
        <img class="compare-plane" id="plane-left" alt=""><img class="compare-plane" id="plane-right" alt="">
        <canvas id="canvas" tabindex="0" aria-label="写真と対応点。左右の矢印キーで点を選択できます。"></canvas>
      </div>
      <p id="point-note"></p>
    </section>
    <dialog id="photo-dialog" class="photo-dialog" aria-labelledby="gallery-title">
      <div class="gallery-head">
        <h2 id="gallery-title">写真を選ぶ</h2>
        <input id="gallery-search" type="search" placeholder="番号・ファイル名" aria-label="写真を絞り込む">
        <button type="button" id="gallery-close">閉じる</button>
      </div>
      <div class="gallery" id="gallery"></div>
    </dialog>
  </main>
  <script>
function matchGraphComponents(nodes) {
  const groups = [];
  const visited = new Set();
  for (const start of nodes) {
    if (visited.has(start)) continue;
    const group = [start];
    visited.add(start);
    for (let i = 0; i < group.length; i++) {
      for (const {node} of group[i].neighbors) {
        if (visited.has(node)) continue;
        visited.add(node);
        group.push(node);
      }
    }
    groups.push(group);
  }
  return groups.sort((a, b) => b.length - a.length);
}

function matchGraphPath(start) {
  if (start.registered) return [start];
  const queue = [start];
  const previous = new Map([[start, null]]);
  for (let i = 0; i < queue.length; i++) {
    const current = queue[i];
    for (const {node} of current.neighbors) {
      if (previous.has(node)) continue;
      previous.set(node, current);
      if (node.registered) {
        const path = [node];
        while (previous.get(path.at(-1))) path.push(previous.get(path.at(-1)));
        return path.reverse();
      }
      queue.push(node);
    }
  }
  return [];
}

// Square thumbnail side 24 world units, 10 units clearance. Spatial hash prevents overlap.
const TILE_SIZE=24, NODE_GAP=34;
function separateNodes(nodes, gap=NODE_GAP) {
  const grid=new Map(), key=(x,y)=>x+","+y;
  function fits(x,y){const ix=Math.floor(x/gap),iy=Math.floor(y/gap);
    for(let i=ix-1;i<=ix+1;i++)for(let j=iy-1;j<=iy+1;j++)for(const p of grid.get(key(i,j))||[])
      if(Math.abs(x-p.x)<gap-1e-7 && Math.abs(y-p.y)<gap-1e-7)return false;
    return true;}
  function insert(p){const k=key(Math.floor(p.x/gap),Math.floor(p.y/gap));if(!grid.has(k))grid.set(k,[]);grid.get(k).push(p);}
  for(const p of nodes){const x=p.x,y=p.y;
    if(!fits(x,y)){let found=false;for(let r=1;!found;r++){
      let best=null,dist=Infinity;
      for(let dx=-r;dx<=r;dx++)for(let dy=-r;dy<=r;dy++){
        if(Math.max(Math.abs(dx),Math.abs(dy))!==r)continue;
        const xx=x+dx*gap/2,yy=y+dy*gap/2,d=dx*dx+dy*dy;
        if(d<dist&&fits(xx,yy)){best=[xx,yy];dist=d;}}
      if(best){[p.x,p.y]=best;found=true;}}
      if(p.vx!==undefined){p.vx*=.2;p.vy*=.2;}}
    insert(p);
  }
}
function placeMatchGraph(groups, edges) {
  const padding = 25;
  const blocks = groups.map(nodes => {
    const members = new Set(nodes);
    const links = edges.filter(edge => members.has(edge.a));
    const iterations = Math.min(144, Math.max(24, Math.floor(520000 / nodes.length ** 2)));
    nodes.forEach((node, i) => {
      const angle = i * 2.399963;
      const radius = Math.sqrt(i) * 23;
      node.x = Math.cos(angle) * radius;
      node.y = Math.sin(angle) * radius;
    });
    for (let step = 0; step < iterations; step++) {
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        a.dx = -a.x * .009;
        a.dy = -a.y * .009;
        for (let j = 0; j < i; j++) {
          const b = nodes[j];
          const dx = b.x - a.x, dy = b.y - a.y;
          const d2 = dx * dx + dy * dy + .01;
          const force = 1960 / (Math.sqrt(d2) * (d2 + 80));
          a.dx -= dx * force;
          a.dy -= dy * force;
          b.dx += dx * force;
          b.dy += dy * force;
        }
      }
      for (const {a, b, count} of links) {
        const dx = b.x - a.x, dy = b.y - a.y;
        const distance = Math.hypot(dx, dy) || 1;
        const length = 62 - Math.min(22, Math.sqrt(count) * .65);
        const force = .049 * (distance - length) / distance;
        a.dx += dx * force;
        a.dy += dy * force;
        b.dx -= dx * force;
        b.dy -= dy * force;
      }
      const cooling = 1 - step / (iterations * 1.3);
      for (const node of nodes) {
        node.x += Math.max(-9, Math.min(9, node.dx)) * cooling;
        node.y += Math.max(-9, Math.min(9, node.dy)) * cooling;
      }
    }
    separateNodes(nodes);
    const minX = Math.min(...nodes.map(node => node.x));
    const minY = Math.min(...nodes.map(node => node.y));
    // Uniform thumbnail sizes; no node-specific shrinking.
    for (const node of nodes) node.tileSize = TILE_SIZE;
    return {
      nodes, minX, minY,
      width: Math.max(...nodes.map(node => node.x)) - minX + padding * 2,
      height: Math.max(...nodes.map(node => node.y)) - minY + padding * 2,
    };
  });
  const budget = Math.max(blocks[0].width + 125,
    Math.sqrt(blocks.reduce((sum, block) => sum + block.width * block.height, 0) * 1.45));
  let x = 0, y = 0, rowHeight = 0, width = 0;
  for (const block of blocks) {
    if (x && x + block.width > budget) {
      x = 0;
      y += rowHeight;
      rowHeight = 0;
    }
    for (const node of block.nodes) {
      node.x += x + padding - block.minX;
      node.y += y + padding - block.minY;
    }
    x += block.width;
    width = Math.max(width, x);
    rowHeight = Math.max(rowHeight, block.height);
  }
  return {width, height: y + rowHeight};
}

function mountMatchGraph(root, index, {fileLabel, registration, photoThumbnail, renderPhotoButton, openPhotoPicker, openPair}) {
  const ui = id => root.querySelector("#" + id);
  const canvas = ui("graph-canvas");
  const context = canvas.getContext("2d");
  const picker = ui("graph-photo");
  const threshold = ui("graph-threshold");
  const display = ui("graph-display");
  const rotation = document.getElementById("rotation-global");
  const countText = n => n.toLocaleString("ja-JP");
  const sequences = new Set(index.photos.map(photo => photo.sequence));
  const uniqueSequences = sequences.size === index.photos.length && !sequences.has(null);
  const ordinal = node => uniqueSequences ? node.sequence : node.id;
  const number = node => "#" + ordinal(node);
  const pairKey = (a, b) => Math.min(a.id, b.id) + "/" + Math.max(a.id, b.id);
  const nodes = index.photos.map(photo => ({...photo, neighbors: []}));
  const nodeById = new Map(nodes.map(node => [node.id, node]));
  const pairs = new Map(index.pairs.map(([a, b, count]) => [pairKey(nodeById.get(a), nodeById.get(b)), count]));
  const count = (a, b) => pairs.get(pairKey(a, b));
  const allEdges = index.pairs.filter(([, , count]) => count > 0)
    .map(([a, b, count]) => ({a: nodeById.get(a), b: nodeById.get(b), count}));
  let edges, groups, extent, selected, path, pathEdges, groupByNode, alternatives;
  let relation = "match", zoom = 1, panX = 0, panY = 0, drag, hover, paintQueued = false;
  const pending = {left: null, right: null};
  const live = ui("graph-live");
  let physicsRAF=0, initialPositions=new Map(), viewCenter=null;
  const fval=id=>Number(ui(id).value);
  function graphScale(){return Math.min((canvas.clientWidth-16)/extent.width,(canvas.clientHeight-16)/extent.height)*zoom;}
  function recenterExtent(){
    const xs=nodes.map(n=>n.x), ys=nodes.map(n=>n.y);
    const minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
    viewCenter=[(minX+maxX)/2,(minY+maxY)/2];
    extent={width:maxX-minX+50,height:maxY-minY+50};
  }
  function physics(){
    physicsRAF=0;if(!live.checked||root.hidden)return;
    const link=fval("force-link")/100, distance=fval("force-distance"), repel=fval("force-repel")/100, center=fval("force-center")/100;
    const [cx,cy]=viewCenter;
    for(const n of nodes){n.fx=-(n.x-cx)*.0025*center*(1+Math.log1p(n.neighbors.length));n.fy=-(n.y-cy)*.0025*center*(1+Math.log1p(n.neighbors.length));n.vx??=0;n.vy??=0;}
    // Exact pair forces for small graphs; bounded sampling for very large runs.
    const stride=Math.max(1,Math.ceil(nodes.length/450));
    for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j+=stride){
      const a=nodes[i],b=nodes[j];let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy);if(d<.001){dx=.01;dy=.01;d=Math.hypot(dx,dy);}
      const f=(2200*repel*stride/(d*d+150))/d;
      a.fx-=dx*f;a.fy-=dy*f;b.fx+=dx*f;b.fy+=dy*f;
    }
    for(const {a,b,count} of edges){let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1;
      const target=Math.max(NODE_GAP,distance/(1+.18*Math.log10(1+count)));
      const strength=.035*link*(1+.35*Math.log10(1+count))/Math.sqrt(Math.max(1,a.neighbors.length,b.neighbors.length));
      const f=(d-target)*strength/d;a.fx+=dx*f;a.fy+=dy*f;b.fx-=dx*f;b.fy-=dy*f;
    }
    for(const n of nodes){if(drag?.node===n){n.vx=n.vy=0;continue;}
      n.vx=Math.max(-9,Math.min(9,(n.vx+n.fx)*.78));n.vy=Math.max(-9,Math.min(9,(n.vy+n.fy)*.78));n.x+=n.vx;n.y+=n.vy;}
    // Gentle continuous collision relaxation first; hard separation guarantees no overlap.
    if(nodes.length<=600)for(let pass=0;pass<3;pass++)for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
      const a=nodes[i],b=nodes[j],dx=b.x-a.x,dy=b.y-a.y,ox=NODE_GAP-Math.abs(dx),oy=NODE_GAP-Math.abs(dy);
      if(ox<=0||oy<=0)continue;const axis=ox<oy?"x":"y",v=axis==="x"?dx:dy,delta=(axis==="x"?ox:oy)+.01,sign=v<0?-1:1;
      if(drag?.node===a)b[axis]+=delta*sign;else if(drag?.node===b)a[axis]-=delta*sign;else{a[axis]-=delta*sign/2;b[axis]+=delta*sign/2;}}
    separateNodes(drag?.node?[drag.node,...nodes.filter(n=>n!==drag.node)]:nodes);
    redraw();physicsRAF=requestAnimationFrame(physics);
  }
  function startPhysics(){if(live.checked&&!root.hidden&&!physicsRAF)physicsRAF=requestAnimationFrame(physics);}
  function stopPhysics(){if(physicsRAF)cancelAnimationFrame(physicsRAF);physicsRAF=0;}
  live.onchange=()=>{ui("force-status").textContent=live.checked?"シミュレーション中（写真をドラッグできます）":"固定表示";if(live.checked)startPhysics();else stopPhysics();};
  for(const id of ["force-link","force-distance","force-repel","force-center"])ui(id).oninput=()=>{ui(id+"-value").textContent=ui(id).value;startPhysics();};
  ui("graph-layout-reset").onclick=()=>{live.checked=false;live.onchange();for(const n of nodes){[n.x,n.y]=initialPositions.get(n);n.vx=n.vy=0;}recenterExtent();resetView();};
  const idleTip = "ホイールで拡大縮小。左クリックで左、右クリックで右。線をクリックして直接比較。";

  function rebuild() {
    for (const node of nodes) node.neighbors = [];
    edges = allEdges.filter(edge => edge.count >= Number(threshold.value));
    for (const {a, b, count} of edges) {
      a.neighbors.push({node: b, count});
      b.neighbors.push({node: a, count});
    }
    for (const node of nodes) node.neighbors.sort((a, b) => b.count - a.count);
    groups = matchGraphComponents(nodes);
    groupByNode = new Map(groups.flatMap(group => group.map(node => [node, group])));
    extent = placeMatchGraph(groups, edges);
    initialPositions=new Map(nodes.map(n=>[n,[n.x,n.y]]));
    for(const n of nodes)n.vx=n.vy=0;
    viewCenter=[extent.width/2,extent.height/2];
    startPhysics();
    ui("graph-count").textContent = countText(edges.length) + "本 ／ " + groups.length + "グループ";
    resetView();
  }

  function resetView() {
    recenterExtent();
    zoom = 1;
    panX = 0;
    panY = 0;
    redraw();
  }

  function pairButton(a, b, label) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.addEventListener("click", () => openPair(a.id, b.id));
    return button;
  }

  function bridgeOptions(start) {
    const group = groupByNode.get(start);
    const weaker = [], untested = [];
    const anchors = groups.filter(other => other !== group && other.some(node => node.registered)).flat();
    for (const a of group) {
      for (const b of anchors) {
        const support = count(a, b);
        if (support > 0) weaker.push({a, b, support});
        if (support === undefined && a.sequence != null && b.sequence != null &&
            a.file.slice(0, a.file.lastIndexOf("/")) === b.file.slice(0, b.file.lastIndexOf("/"))) {
          untested.push({a, b, distance: Math.abs(a.sequence - b.sequence)});
        }
      }
    }
    weaker.sort((a, b) => b.support - a.support);
    untested.sort((a, b) => a.distance - b.distance);
    return weaker.slice(0, 2).concat(untested.slice(0, 2));
  }

  function renderPath() {
    const panel = ui("graph-path");
    panel.replaceChildren();
    panel.hidden = selected.registered;
    if (selected.registered) return;
    const header = document.createElement("h2");
    header.textContent = "登録済みへの経路";
    const note = document.createElement("p");
    panel.append(header, note);
    if (!path.length) {
      note.textContent = "表示中の線では登録済みにつながりません。";
      const options = bridgeOptions(selected);
      if (options.length) {
        const caption = document.createElement("p");
        caption.textContent = "登録済みを含むグループとの照合候補";
        panel.append(caption);
      }
      for (const {a, b, support} of options) {
        const label = support ? countText(support) + "点・表示条件未満" : "未照合・撮影順が近い";
        panel.append(pairButton(a, b, number(a) + " ↔ " + number(b) + "　" + label));
      }
      return;
    }
    note.textContent = path.map(number).join(" → ") + "　／　最短 " + (path.length - 1) + " 本";
    for (let i = 1; i < path.length; i++) {
      const a = path[i - 1], b = path[i];
      panel.append(pairButton(a, b, number(a) + " ↔ " + number(b) + "　" + countText(count(a, b)) + "点"));
    }
  }

  function renderRelations() {
    const categories = {match: [], zero: [], unseen: []};
    for (const b of nodes) {
      if (b === selected) continue;
      const support = count(selected, b);
      categories[support === undefined ? "unseen" : support ? "match" : "zero"].push({node: b, support});
    }
    categories.match.sort((a, b) => b.support - a.support);
    const related = categories[relation];
    alternatives = relation === "match" ? [] : related.map(({node}) => ({a: selected, b: node, count: 0}));
    const labels = {match: "一致", zero: "照合0点", unseen: "未照合"};
    for (const button of root.querySelectorAll("[data-relation]")) {
      const key = button.dataset.relation;
      button.textContent = labels[key] + " " + categories[key].length;
      button.setAttribute("aria-pressed", key === relation);
    }
    const listing = ui("graph-relations");
    listing.replaceChildren(...related.map(({node, support}) => {
      const suffix = support === undefined ? "未照合" : countText(support) + "点";
      const button = pairButton(selected, node, "");
      button.className = "photo-choice";
      renderPhotoButton(button, node, suffix + " ／ " + registration(node),
        number(node) + " " + fileLabel(node), true);
      return button;
    }));
    if (!related.length) {
      const empty = document.createElement("p");
      empty.textContent = "該当する組はありません。";
      listing.append(empty);
    }
    listing.scrollTop = 0;
    ui("graph-tip").textContent = defaultTip();
  }

  function defaultTip() {
    if (relation === "zero") return "灰色の破線は照合0点です。";
    if (relation === "unseen") return "青い点線は未照合です。";
    return idleTip;
  }

  function select(node, keepRelation = false) {
    selected = node;
    if (!keepRelation) relation = "match";
    renderPhotoButton(picker, node, registration(node), number(node) + " " + fileLabel(node));
    path = matchGraphPath(node);
    pathEdges = new Set(path.slice(1).map((target, i) => pairKey(path[i], target)));
    ui("node-status").textContent = (node.registered ? "登録済み" : "未登録") +
      " ／ 表示中の線 " + node.neighbors.length + " 本" + " ／ グループ " + groupByNode.get(node).length + " 枚";
    renderPath();
    renderRelations();
    redraw();
  }

  function showPending() {
    const status = ui("graph-pending");
    status.hidden = ui("graph-pick-clear").hidden = !pending.left && !pending.right;
    status.textContent = "左 " + (pending.left ? number(pending.left) : "未選択") +
      " ／ 右 " + (pending.right ? number(pending.right) : "未選択");
    redraw();
  }

  function clearPending() {
    pending.left = pending.right = null;
    showPending();
  }

  function chooseSide(side, node) {
    const other = side === "left" ? "right" : "left";
    if (pending[other] === node) pending[other] = null;
    pending[side] = node;
    if (pending.left && pending.right) {
      const {left, right} = pending;
      clearPending();
      openPair(left.id, right.id);
    } else {
      showPending();
    }
  }

  function redraw() {
    if (paintQueued) return;
    paintQueued = true;
    requestAnimationFrame(() => {
      paintQueued = false;
      paint();
    });
  }

  function paint() {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height || !extent) return;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const scale = Math.min((width - 16) / extent.width, (height - 16) / extent.height) * zoom;
    const showPhotos = display.value === "photos";
    for (const node of nodes) {
      node.screenX = width / 2 + (node.x - viewCenter[0]) * scale + panX;
      node.screenY = height / 2 + (node.y - viewCenter[1]) * scale + panY;
      node.radius = showPhotos ? scale * node.tileSize / 2
        : Math.max(4, Math.min(12, scale * (8 + Math.sqrt(node.neighbors.length)) + 2));
    }
    for (const edge of edges) {
      const {a, b, count} = edge;
      const nearby = a === selected || b === selected || a === hover || b === hover;
      const color = nearby ? "#99baafd0" : "#789b933d";
      wire(edge, color, (nearby ? 1.2 : .55) + Math.min(2.5, Math.sqrt(count) / 16));
    }
    for (const edge of edges) {
      if (pathEdges.has(pairKey(edge.a, edge.b))) wire(edge, "#ffe29a", 2.6);
    }
    context.setLineDash(relation === "zero" ? [5, 4] : [1, 5]);
    for (const edge of alternatives) wire(edge, relation === "zero" ? "#c5d1cf75" : "#8cbbff8a", 1.1);
    context.setLineDash([]);
    if (hover?.a) wire(hover, "#f5ffdc", 2.8);
    for (const node of nodes) drawNode(node, showPhotos);
  }

  function drawNode(node, showPhotos) {
    const {screenX: x, screenY: y, radius} = node;
    const color = node.registered ? "#8fdfc5" : "#ffc17a";
    const active = node === selected || node === hover || path.includes(node);
    const selectedForPair = node === pending.left || node === pending.right;
    if (showPhotos) {
      if (!node.preview) {
        node.preview = photoThumbnail(node, false, true);
        node.preview.addEventListener("load", redraw, {once: true});
      }
      const image = node.preview;
      const edge = radius * 2;
      context.fillStyle = "#13292d";
      context.fillRect(x - radius, y - radius, edge, edge);
      if (image.naturalWidth) {
        let degrees = rotation.value === "off" ? 0 : (node.roll ?? 0);
        if (rotation.value === "90") degrees = Math.round(degrees / 90) * 90;
        const angle = degrees * Math.PI / 180;
        const c = Math.abs(Math.cos(angle)), s = Math.abs(Math.sin(angle));
        const w = image.naturalWidth, h = image.naturalHeight;
        const scale = Math.max(.0001, edge - Math.min(2,edge*.1)) / Math.max(c*w+s*h, s*w+c*h);
        context.save(); context.translate(x,y); context.rotate(angle);
        context.drawImage(image, -w*scale/2, -h*scale/2, w*scale, h*scale);
        context.restore();
      } else {
        drawLabel(node, color);
      }
      if (!node.neighbors.length) context.setLineDash([3, 2]);
      context.strokeStyle = color;
      context.lineWidth = 1.6;
      context.strokeRect(x - radius, y - radius, edge, edge);
      context.setLineDash([]);
    } else {
      drawRing(node, radius);
      context.fillStyle = node.neighbors.length ? color : "#183237";
      context.fill();
      context.strokeStyle = color;
      context.lineWidth = node.neighbors.length ? 1 : 1.6;
      context.stroke();
      if (radius >= 7 || active || selectedForPair) drawLabel(node, node.neighbors.length ? "#102e2b" : color);
    }
    if (active || selectedForPair) {
      context.strokeStyle = node === pending.left ? "#caffe6" : node === pending.right ? "#bad8ff"
        : node === selected ? "#f8fff4" : "#ffe29a";
      context.lineWidth = node === selected || selectedForPair ? 2 : 1;
      const outer = radius + Math.min(3,radius*.15);
      if (showPhotos) {
        context.strokeRect(x - outer, y - outer, outer * 2, outer * 2);
      } else {
        drawRing(node, outer);
        context.stroke();
      }
    }
  }

  function drawRing(node, radius) {
    context.beginPath();
    context.arc(node.screenX, node.screenY, radius, 0, Math.PI * 2);
  }

  function drawLabel(node, color) {
    context.fillStyle = color;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.font = "600 " + Math.min(11, node.radius + 1) + "px sans-serif";
    context.fillText(ordinal(node), node.screenX, node.screenY + .5, node.radius * 1.8);
  }

  function wire({a, b}, color, weight) {
    context.strokeStyle = color;
    context.lineWidth = weight;
    context.beginPath();
    context.moveTo(a.screenX, a.screenY);
    context.lineTo(b.screenX, b.screenY);
    context.stroke();
  }

  function hit(event, nodesOnly = false) {
    const bounds = canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
    const showPhotos = display.value === "photos";
    const nearby = nodes.filter(node => {
      const dx = Math.abs(x - node.screenX), dy = Math.abs(y - node.screenY);
      return showPhotos ? Math.max(dx, dy) < node.radius + 3 : Math.hypot(dx, dy) < node.radius + 5;
    })
      .sort((a, b) => Math.hypot(x - a.screenX, y - a.screenY) - Math.hypot(x - b.screenX, y - b.screenY));
    if (nearby.length || nodesOnly) return nearby[0];
    let closest, maxDistance = 5;
    for (const edge of edges.concat(alternatives)) {
      const {a, b} = edge;
      const dx = b.screenX - a.screenX, dy = b.screenY - a.screenY;
      const length = dx * dx + dy * dy;
      const place = Math.max(0, Math.min(1, ((x - a.screenX) * dx + (y - a.screenY) * dy) / length));
      const distance = Math.hypot(x - a.screenX - place * dx, y - a.screenY - place * dy);
      if (distance < maxDistance) {
        closest = edge;
        maxDistance = distance;
      }
    }
    return closest;
  }

  canvas.addEventListener("pointermove", event => {
    if (drag) {
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (drag.moved || Math.hypot(dx, dy) > 4) {
        drag.moved = true;
        if(drag.node){drag.node.x=drag.nx+dx/graphScale();drag.node.y=drag.ny+dy/graphScale();separateNodes([drag.node,...nodes.filter(n=>n!==drag.node)]);}
        else {panX = drag.panX + dx; panY = drag.panY + dy;}
        canvas.style.cursor = "grabbing";
        redraw();
      }
      return;
    }
    const target = hit(event);
    if (target === hover) return;
    hover = target;
    canvas.style.cursor = target ? "pointer" : "grab";
    ui("graph-tip").textContent = target?.a ?
      number(target.a) + " ↔ " + number(target.b) + "　" +
        (count(target.a, target.b) === undefined ? "未照合" : countText(count(target.a, target.b)) + "点") :
      target ? number(target) + " " + fileLabel(target) + "　" +
        (target.registered ? "登録済み" : "未登録") : defaultTip();
    redraw();
  });
  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;
    const node=live.checked?hit(event,true):null;
    drag = {x: event.clientX, y: event.clientY, panX, panY, moved: false,node,nx:node?.x,ny:node?.y};
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointerup", event => {
    if (!drag) return;
    const moved = drag.moved;
    drag = null;
    canvas.style.cursor = "grab";
    if (moved) return;
    const target = hit(event);
    if (target?.a) openPair(target.a.id, target.b.id);
    else if (target) {
      select(target);
      chooseSide("left", target);
    }
  });
  canvas.addEventListener("contextmenu", event => {
    const node = hit(event, true);
    if (!node) return;
    event.preventDefault();
    if (!pending.left) select(node);
    chooseSide("right", node);
  });
  canvas.addEventListener("pointercancel", () => { drag = null; canvas.style.cursor = "grab"; });
  canvas.addEventListener("pointerleave", () => { hover = null; ui("graph-tip").textContent = defaultTip(); redraw(); });
  picker.addEventListener("click", () => openPhotoPicker({
    title: "グラフの写真を選ぶ", choices: nodes, selected: selected.id,
    onSelect: id => select(nodeById.get(id)),
  }));
  ui("graph-pick-left").addEventListener("click", () => chooseSide("left", selected));
  ui("graph-pick-right").addEventListener("click", () => chooseSide("right", selected));
  ui("graph-pick-clear").addEventListener("click", clearPending);
  display.addEventListener("change", redraw);
  rotation.addEventListener("change", redraw);
  canvas.addEventListener("wheel", event => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    const x = event.clientX - bounds.left - bounds.width / 2;
    const y = event.clientY - bounds.top - bounds.height / 2;
    const old = zoom;
    const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? bounds.height : 1);
    zoom = Math.max(.2, Math.min(20, zoom * Math.exp(-delta * .0015)));
    panX = x - (x - panX) * zoom / old;
    panY = y - (y - panY) * zoom / old;
    redraw();
  }, {passive:false});
  threshold.addEventListener("change", () => { rebuild(); select(selected, true); });
  for (const button of root.querySelectorAll("[data-relation]")) {
    button.addEventListener("click", () => { relation = button.dataset.relation; renderRelations(); redraw(); });
  }
  ui("graph-reset").addEventListener("click", resetView);
  ui("graph-in").addEventListener("click", () => { zoom = Math.min(20, zoom * 1.35); redraw(); });
  ui("graph-out").addEventListener("click", () => { zoom = Math.max(.2, zoom / 1.35); redraw(); });
  new ResizeObserver(redraw).observe(canvas);
  rebuild();
  select(nodes.find(node => !node.registered && matchGraphPath(node).length > 2) || nodes[0]);
  return {redraw(){redraw();startPhysics();},pause:stopPhysics};
}

  </script>
  <script>
    const byId = id => document.getElementById(id);
    const format = n => n.toLocaleString("ja-JP");
    const canvas = byId("canvas");
    const context = canvas.getContext("2d");
    const limit = byId("limit");
    const lines = byId("lines");
    const sides = ["left", "right"].map(id => ({id, button: byId(id), turn: 0}));
    const dialog = byId("photo-dialog");
    const gallery = byId("gallery");
    const search = byId("gallery-search");
    let photos, photosById, pairIndex, repeatedNames, graph;
    let pickerRequest;
    let points = [], displayed = [], drawn = [];
    let hovered = -1, pinned = -1, request = 0, paintQueued = false;
    let fit = null, aligned = false, mode = "side", zoom = 1, pan = [0,0], orbit = [0,0], drag = null, suppressClick = false;
    const D = Math.PI / 180, F = 2.5, sliders = ["rx","ry","rz","tx","ty","tz","sc","opacityA","opacityB"];
    const val = id => Number(byId(id).value);
    function resetCompare(resetTurns = true) {
      zoom=1; pan=[0,0]; orbit=[0,0]; aligned=false;
      if(resetTurns) for(const side of sides) side.turn=0;
      for(const id of sliders) byId(id).value=id==="sc"?100:id.startsWith("opacity")?60:0;
      updateTransformUI(); schedulePaint();
    }
    function updateTransformUI() {
      for(const id of sliders) byId(id+"v").textContent=val(id)+(id.startsWith("r")?"°":id==="sc"||id.startsWith("opacity")?"%":"");
      byId("auto-align").disabled=!fit;
      byId("auto-align").setAttribute("aria-pressed",aligned);
      byId("compare-zoom").textContent=Math.round(zoom*100)+"%";
      byId("fit-info").textContent=fit ? (aligned ? `位置合わせ: ${fit.inliers}/${fit.total}点・中央誤差 ${fit.median_px.toFixed(1)}px` : `推定相対回転 ${fit.angle.toFixed(1)}°`) : "";
    }
    function angle(i){
      let deg=0; const kind=byId("rotation-global").value;
      if(kind!=="off") {deg=sides[i].photo?.roll ?? 0;if(kind==="90")deg=Math.round(deg/90)*90;}
      return deg*D+sides[i].turn*Math.PI/2;
    }
    function size(i){const image=sides[i].image;const w=image?.naturalWidth||1,h=image?.naturalHeight||1;return [w,h,Math.max(w,h)];}
    function norm(i,x,y){const [w,h,m]=size(i);return [(x-w/2)/m,(y-h/2)/m];}
    function rot(p,x,y,z){let [a,b,c]=p;[b,c]=[b*Math.cos(x)-c*Math.sin(x),b*Math.sin(x)+c*Math.cos(x)];[a,c]=[a*Math.cos(y)+c*Math.sin(y),-a*Math.sin(y)+c*Math.cos(y)];return [a*Math.cos(z)-b*Math.sin(z),a*Math.sin(z)+b*Math.cos(z),c];}
    function base(i,x,y){
      if(i===0)return rot([x,y,0],0,0,angle(0));
      if(aligned&&fit){const h=fit.matrix;const p=rot([x,y,0],0,0,sides[1].turn*Math.PI/2);return rot([h[0][0]*p[0]+h[0][1]*p[1]+h[0][2],h[1][0]*p[0]+h[1][1]*p[1]+h[1][2],F*(h[2][0]*p[0]+h[2][1]*p[1]+h[2][2]-1)],0,0,angle(0));}
      return rot([x,y,.10],0,0,angle(1));
    }
    function world(i,x,y){let p=base(i,x,y);if(i===1){const c=base(i,0,0),s=val("sc")/100;const q=rot(p.map((v,j)=>(v-c[j])*s),val("rx")*D,val("ry")*D,val("rz")*D);p=q.map((v,j)=>v+c[j]+val(["tx","ty","tz"][j])/100);}return rot(p,orbit[0],orbit[1],0);}
    function project(i,x,y,W,H){const p=world(i,x,y),den=F+p[2];if(den<.12)return null;const s=Math.min(W,H)*.8*zoom*F/den;return [W/2+pan[0]+p[0]*s,H/2+pan[1]+p[1]*s];}
    function solve(m){let n=m.length;for(let i=0;i<n;i++){let best=i;for(let j=i+1;j<n;j++)if(Math.abs(m[j][i])>Math.abs(m[best][i]))best=j;[m[i],m[best]]=[m[best],m[i]];const p=m[i][i];if(Math.abs(p)<1e-10)return null;for(let k=i;k<=n;k++)m[i][k]/=p;for(let j=0;j<n;j++){if(j===i)continue;let f=m[j][i];for(let k=i;k<=n;k++)m[j][k]-=f*m[i][k];}}return m.map(r=>r[n]);}
    function cssMatrix(src,dst){const a=[];for(let i=0;i<4;i++){const [x,y]=src[i],[u,v]=dst[i];a.push([x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]);}const q=solve(a);if(!q||q.some(v=>!Number.isFinite(v)))return null;const [a0,b,c,d,e,f,g,h]=q;return `matrix3d(${[a0,d,0,g,b,e,0,h,0,0,1,0,c,f,0,1].join(',')})`;}
    function paintPlane(i,W,H){
      const side=sides[i], im=byId("plane-"+side.id); im.style.display="none";
      if(!side.image?.naturalWidth)return null;
      const [w,h,m]=size(i),iw=w/m*1000,ih=h/m*1000;
      const dst=[[-w/2/m,-h/2/m],[w/2/m,-h/2/m],[w/2/m,h/2/m],[-w/2/m,h/2/m]].map(p=>project(i,...p,W,H));
      if(dst.some(p=>!p))return null;
      const tr=cssMatrix([[0,0],[iw,0],[iw,ih],[0,ih]],dst);if(!tr)return null;
      if(im.getAttribute("src")!==side.image.src)im.src=side.image.src;
      im.style.width=iw+"px";im.style.height=ih+"px";im.style.transform=tr;im.style.opacity=val(i?"opacityB":"opacityA")/100;im.style.display="block";im.style.zIndex=world(i,0,0)[2]>world(1-i,0,0)[2]?1:2;
      return (x,y)=>project(i,...norm(i,x,y),W,H);
    }

    const pairKey = (a, b) => Math.min(a, b) + "/" + Math.max(a, b);
    const pairCount = (a, b) => pairIndex.get(pairKey(a, b));
    const basename = photo => photo.file.split("/").pop();
    const fileLabel = photo => repeatedNames.has(basename(photo)) ? photo.file : basename(photo);
    const registration = photo => photo.registered ? "登録済み" : "未登録";

    function photoThumbnail(photo, lazy = false, raw = false) {
      const image = document.createElement("img");
      image.className = "photo-thumbnail";
      image.alt = "";
      image.title = photo.roll==null?"地面基準の回転不明":`地面基準 ${photo.roll.toFixed(1)}° / ${photo.orientation_source||""}`;
      image.decoding = "async";
      if (lazy) image.loading = "lazy";
      if (photo.available) {
        image.addEventListener("error", () => {
          image.classList.add("unavailable");
          image.removeAttribute("src");
        }, {once: true});
        if(!raw)image.dataset.photoId=photo.id;
        image.src = "./api/thumb/" + photo.id + (raw?"":"?rotation="+byId("rotation-global").value);
      } else {
        image.classList.add("unavailable");
      }
      return image;
    }

    function renderPhotoButton(button, photo, detail = registration(photo), label = fileLabel(photo), lazy = false) {
      const text = document.createElement("span");
      text.className = "photo-label";
      const name = document.createElement("span");
      name.className = "photo-name";
      name.textContent = label;
      name.title = photo.file;
      const meta = document.createElement("span");
      meta.className = "photo-detail";
      meta.textContent = detail;
      text.append(name, meta);
      button.replaceChildren(photoThumbnail(photo, lazy), text);
    }

    function renderGallery() {
      const {choices, selected, describe, onSelect} = pickerRequest;
      const query = search.value.trim().toLocaleLowerCase();
      const filtered = choices.filter(photo =>
        (photo.file + " #" + (photo.sequence ?? photo.id)).toLocaleLowerCase().includes(query)
      );
      gallery.replaceChildren(...filtered.map(photo => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "gallery-card";
        button.setAttribute("aria-pressed", photo.id === selected);
        renderPhotoButton(button, photo, describe(photo), fileLabel(photo), true);
        button.addEventListener("click", () => {
          dialog.close();
          onSelect(photo.id);
        });
        return button;
      }));
      if (!filtered.length) {
        const empty = document.createElement("p");
        empty.textContent = "該当する写真がありません。";
        gallery.append(empty);
      }
    }

    function openPhotoPicker({title, choices, selected, describe = registration, onSelect}) {
      pickerRequest = {choices, selected, describe, onSelect};
      byId("gallery-title").textContent = title;
      search.value = "";
      renderGallery();
      dialog.showModal();
      dialog.scrollTop = 0;
      search.focus();
    }

    search.addEventListener("input", renderGallery);
    byId("gallery-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => {
      if (dialog.open) return;
      pickerRequest = null;
      gallery.replaceChildren();
    });

    async function getJSON(path) {
      const response = await fetch(path);
      if (!response.ok) {let message;try{message=(await response.json()).error;}catch{}throw new Error(message||"照合データを読み込めませんでした。");}
      return response.json();
    }

    function schedulePaint() {
      if (paintQueued) return;
      paintQueued = true;
      requestAnimationFrame(() => {
        paintQueued = false;
        paint();
      });
    }

    function loadPhoto(side, id) {
      if (side.photo?.id === id) return;
      side.photo = photosById.get(id);
      side.turn = 0;
      side.error = "";
      byId(side.id + "-features").textContent = "特徴点 " + format(side.photo.features)+" ／ 地面基準 "+(side.photo.roll==null?"不明":side.photo.roll.toFixed(1)+"°");
      const image = new Image();
      side.image = image;
      image.onload = schedulePaint;
      image.onerror = () => {
        if (side.image !== image) return;
        side.error = side.photo.available ? "元写真を表示できません" : "元写真が見つかりません";
        schedulePaint();
      };
      image.src = "./api/photo/" + id;
      schedulePaint();
    }

    function matchDetail(photo, left) {
      const count = pairCount(left, photo.id);
      return registration(photo) + " ／ " + (count === undefined ? "未照合" : format(count) + "点");
    }

    function rightPhotos(left) {
      return photos.filter(photo => photo.id !== left).sort((a, b) =>
        (pairCount(left, b.id) ?? -1) - (pairCount(left, a.id) ?? -1)
      );
    }

    async function compare(left, right) {
      const revision = ++request;
      if (left === right || !photosById.has(right)) right = rightPhotos(left)[0].id;
      loadPhoto(sides[0], left);
      loadPhoto(sides[1], right);
      renderPhotoButton(sides[0].button, sides[0].photo);
      renderPhotoButton(sides[1].button, sides[1].photo, matchDetail(sides[1].photo, left));
      for (const button of byId("examples").children) {
        button.setAttribute("aria-pressed", button.dataset.pair === pairKey(left, right));
      }
      points = []; fit=null; resetCompare();
      byId("pair-status").textContent = "読み込み中…";
      byId("pair-support").textContent = "";
      refreshPoints();
      try {
        const result = await getJSON("./api/pair/" + left + "/" + right);
        if (revision !== request) return;
        points = result.points;
        byId("pair-status").textContent = result.tested ? format(points.length) + " 点の対応候補" : "この組は照合対象外";
        const registered = sides.filter(side => side.photo.registered).length;
        if (points.length && registered) {
          const label = registered === 2 ? "同じ3D点に登録" : "登録済み側で3D点あり";
          byId("pair-support").textContent = label + " " + format(result.structure_matches) + " 点";
        }
        refreshPoints();
        if(points.length>=2) {
          try { const res=await getJSON("./api/alignment/"+left+"/"+right); if(revision!==request)return; fit=res.alignment; updateTransformUI(); schedulePaint(); }
          catch(error){ if(revision===request) byId("fit-info").textContent="自動回転・位置合わせの推定に失敗"; }
        }
      } catch (error) {
        if (revision !== request) return;
        byId("pair-status").textContent = error.message;
      }
    }

    function refreshPoints() {
      const max = limit.value === "all" ? points.length : Math.min(Number(limit.value), points.length);
      displayed = Array.from({length: max}, (_, i) => points[Math.floor(i * points.length / max)]);
      pinned = -1;
      hovered = -1;
      drawn = [];
      limit.disabled = !points.length;
      lines.disabled = !points.length;
      byId("display-count").textContent = points.length ? format(max) + " / " + format(points.length) + " 点を表示" : "";
      updateSelection();
    }

    function activePoint() {
      return pinned >= 0 ? pinned : hovered;
    }

    function updateSelection() {
      const point = displayed[activePoint()];
      byId("point-note").textContent = point ?
        "左 (" + point[0] + ", " + point[1] + ")　右 (" + point[2] + ", " + point[3] + ") px" :
        "点をクリックで固定。ホイールで拡大縮小、ドラッグで移動（3Dでは左ドラッグで視点回転、右ドラッグで移動）。";
      schedulePaint();
    }

    function angleForSide(i){return angle(i);}
    function paintPhoto(side, x, width, height, inset) {
      const image = side.image;
      const cx = x + width / 2 + pan[0];
      const cy = height / 2 + pan[1];
      if (!image?.naturalWidth) {
        context.fillStyle = "#d4e5df";
        context.font = "14px sans-serif";
        context.textAlign = "center";
        context.fillText(side.error || "写真を読み込み中…", cx, cy, width - inset * 2);
        return null;
      }
      const w = image.naturalWidth;
      const h = image.naturalHeight;
      const angle = side===sides[0] ? angleForSide(0) : angleForSide(1);
      const base = Math.min(...sides.map((s,i)=>{const [iw,ih,m]=size(i),a=angleForSide(i),c=Math.abs(Math.cos(a)),sn=Math.abs(Math.sin(a));return Math.min((width-inset*2)*m/(c*iw+sn*ih),(height-inset*2)*m/(sn*iw+c*ih));}));
      const scale=base*zoom/Math.max(w,h);
      context.save();
      context.translate(cx, cy);
      context.rotate(angle);
      context.drawImage(image, -w * scale / 2, -h * scale / 2, w * scale, h * scale);
      context.restore();
      const cos = Math.cos(angle), sin = Math.sin(angle);
      return (rawX, rawY) => {
        const dx = (rawX - w / 2) * scale, dy = (rawY - h / 2) * scale;
        return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos];
      };
    }

    function drawMatch(pair, color, dotRadius, lineWidth) {
      if (lineWidth) {
        context.strokeStyle = color;
        context.lineWidth = lineWidth;
        context.beginPath();
        context.moveTo(...pair[0]);
        context.lineTo(...pair[1]);
        context.stroke();
      }
      context.fillStyle = color;
      context.strokeStyle = "#132b2e";
      context.lineWidth = 1;
      for (const [x, y] of pair) {
        context.beginPath();
        context.arc(x, y, dotRadius, 0, Math.PI * 2);
        context.fill();
        context.stroke();
      }
    }

    function paint() {
      const width = canvas.clientWidth, height = canvas.clientHeight;
      if (!width || !height) return;
      const pixelRatio = window.devicePixelRatio || 1;
      const pixelsWide = Math.round(width * pixelRatio), pixelsHigh = Math.round(height * pixelRatio);
      if (canvas.width !== pixelsWide || canvas.height !== pixelsHigh) {
        canvas.width = pixelsWide;
        canvas.height = pixelsHigh;
      }
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);
      let left,right;
      if(mode==="side") {
        byId("plane-left").style.display=byId("plane-right").style.display="none";
        context.fillStyle="#516c6d"; context.fillRect(width/2,10,1,height-20);
        const inset=width<520?6:12;
        context.save(); context.beginPath(); context.rect(0,0,width/2,height); context.clip(); left=paintPhoto(sides[0],0,width/2,height,inset); context.restore();
        context.save(); context.beginPath(); context.rect(width/2,0,width/2,height); context.clip(); right=paintPhoto(sides[1],width/2,width/2,height,inset); context.restore();
      } else {left=paintPlane(0,width,height);right=paintPlane(1,width,height);}
      drawn = left && right ? displayed.map(([x,y,u,v])=>[left(x,y),right(u,v)]) : [];
      if(mode==="side") drawn=drawn.map(pair=>pair[0][0]<0||pair[0][0]>width/2||pair[1][0]<width/2||pair[1][0]>width||pair.some(p=>p[1]<0||p[1]>height)?null:pair);
      else drawn=drawn.map(pair=>pair.every(Boolean)?pair:null);
      for (const pair of drawn) {
        if (!pair) continue;
        if (lines.checked) {
          context.strokeStyle = drawn.length > 120 ? "#7cebd347" : "#7cebd37d";
          context.lineWidth = .8;
          context.beginPath();
          context.moveTo(...pair[0]);
          context.lineTo(...pair[1]);
          context.stroke();
        }
      }
      if(byId("dots").checked) for (const pair of drawn) if(pair) drawMatch(pair, "#93f4df", drawn.length > 120 ? 2 : 3, 0);
      if (drawn[activePoint()]) drawMatch(drawn[activePoint()], "#f8ff9a", 5, 2);
    }

    function nearbyPoint(event) {
      const bounds = canvas.getBoundingClientRect();
      const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
      let closest = -1, distance = event.pointerType === "touch" ? 196 : 100;
      drawn.forEach((pair, i) => {
        if(!pair)return;
        for (const [px, py] of pair) {
          const squared = (px - x) ** 2 + (py - y) ** 2;
          if (squared < distance) {
            distance = squared;
            closest = i;
          }
        }
      });
      return closest;
    }

    canvas.addEventListener("pointermove", event => {
      if(drag)return;
      const point = nearbyPoint(event);
      canvas.style.cursor = point >= 0 ? "crosshair" : "";
      if (point !== hovered) {
        hovered = point;
        updateSelection();
      }
    });
    canvas.addEventListener("pointerleave", () => { hovered = -1; updateSelection(); });
    canvas.addEventListener("click", event => {
      if(suppressClick){suppressClick=false;return;}
      const point = nearbyPoint(event);
      pinned = point === pinned ? -1 : point;
      updateSelection();
    });
    canvas.addEventListener("keydown", event => {
      if (event.key === "Escape") {
        hovered = -1;
        pinned = -1;
      } else if (displayed.length && ["ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        const direction = event.key === "ArrowLeft" ? -1 : 1;
        pinned = pinned < 0 ? 0 : (pinned + direction + displayed.length) % displayed.length;
      } else {
        return;
      }
      updateSelection();
    });
    for (const side of sides) {
      byId(side.id + "-rotate").addEventListener("click", () => {
        side.turn = (side.turn + 1) % 4;
        schedulePaint();
      });
    }
    function setMode(next){mode=next; zoom=1;pan=[0,0];orbit=[0,0];byId("mode-side").setAttribute("aria-pressed",mode==="side");byId("mode-three").setAttribute("aria-pressed",mode==="three");byId("three-controls").hidden=mode!=="three";updateTransformUI();schedulePaint();}
    byId("mode-side").onclick=()=>setMode("side");byId("mode-three").onclick=()=>setMode("three");
    byId("compare-reset").onclick=()=>resetCompare();
    byId("front-view").onclick=()=>{orbit=[0,0];schedulePaint();};
    byId("auto-align").onclick=()=>{if(!fit)return;resetCompare();aligned=true;updateTransformUI();schedulePaint();};
    byId("rotation-global").onchange=()=>{
      aligned=false;updateTransformUI();schedulePaint();
      for(const im of document.querySelectorAll("img[data-photo-id]"))im.src="./api/thumb/"+im.dataset.photoId+"?rotation="+byId("rotation-global").value;
      try{localStorage.setItem("feature-match-rotation",byId("rotation-global").value);}catch{}
    };
    byId("dots").onchange=schedulePaint;
    for(const id of sliders)byId(id).oninput=()=>{updateTransformUI();schedulePaint();};
    canvas.addEventListener("wheel",e=>{e.preventDefault();const old=zoom,r=canvas.getBoundingClientRect();const x=e.clientX-r.left,y=e.clientY-r.top;zoom=Math.max(.15,Math.min(15,zoom*Math.exp(-e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?r.height:1)*.0015)));const cx=r.width*(mode==="side"?(x<r.width/2?.25:.75):.5);pan=[x-cx-(x-cx-pan[0])*zoom/old,y-r.height/2-(y-r.height/2-pan[1])*zoom/old];updateTransformUI();schedulePaint();},{passive:false});
    canvas.addEventListener("pointerdown",e=>{if(e.button!==0&&e.button!==2)return;suppressClick=false;drag={x:e.clientX,y:e.clientY,move:mode==="side"||e.button===2||e.shiftKey,moved:false};canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener("pointermove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(!drag.moved&&Math.hypot(dx,dy)<4)return;drag.moved=true;drag.x=e.clientX;drag.y=e.clientY;canvas.style.cursor="grabbing";if(drag.move){pan[0]+=dx;pan[1]+=dy;}else{orbit[0]+=dy*.008;orbit[1]+=dx*.008;}schedulePaint();});
    for(const event of ["pointerup","pointercancel","lostpointercapture"])canvas.addEventListener(event,()=>{if(!drag)return;suppressClick=drag.moved;drag=null;canvas.style.cursor="";});
    canvas.addEventListener("contextmenu",e=>e.preventDefault());
    limit.addEventListener("change", refreshPoints);
    lines.addEventListener("change", schedulePaint);
    new ResizeObserver(schedulePaint).observe(canvas);

    function switchView(showGraph) {
      byId("graph-view").hidden = !showGraph;
      byId("comparison").hidden = showGraph;
      byId("show-graph").setAttribute("aria-pressed", showGraph);
      byId("show-comparison").setAttribute("aria-pressed", !showGraph);
      if (showGraph) graph.redraw();
      else {graph.pause();schedulePaint();}
    }

    async function start() {
      try{const saved=localStorage.getItem("feature-match-rotation");if(["90","free","off"].includes(saved))byId("rotation-global").value=saved;}catch{}
      const runList=await getJSON("./api/runs");
      const chooser=byId("run-select");
      for(const name of runList.runs){const op=document.createElement("option");op.value=name;op.textContent=name;chooser.append(op);}
      chooser.value=runList.selected;chooser.onchange=()=>{location.href="/run/"+encodeURIComponent(chooser.value)+"/";};
      const index = await getJSON("./api/index");
      photos = index.photos;
      byId("orientation-summary").textContent=`地面基準の回転: ${photos.filter(p=>p.roll!=null).length}/${photos.length}枚（不明は補正なし）`;
      photosById = new Map(photos.map(photo => [photo.id, photo]));
      const names = new Set();
      repeatedNames = new Set();
      for (const photo of photos) {
        const name = basename(photo);
        if (names.has(name)) repeatedNames.add(name);
        names.add(name);
      }
      pairIndex = new Map(index.pairs.map(([a, b, count]) => [pairKey(a, b), count]));
      const matches = index.pairs.filter(([, , count]) => count > 0)
        .sort((a, b) => b[2] - a[2]);
      const available = ([a, b]) => photosById.get(a).available && photosById.get(b).available;
      const registered = id => photosById.get(id).registered;
      const usable = matches.filter(available);
      const mixed = usable.find(([a, b]) => registered(a) !== registered(b));
      const unregistered = usable.find(([a, b]) => !registered(a) && !registered(b));
      const both = usable.find(([a, b]) => registered(a) && registered(b));
      const zero = index.pairs.find(pair => pair[2] === 0 && available(pair));
      byId("summary").textContent = photos.length + "枚中 " + photos.filter(photo => photo.registered).length
        + "枚登録　／　" + format(matches.length) + "組で一致　／　" + index.run;
      for (const [label, example] of [["片方が未登録", mixed], ["両方が未登録", unregistered],
                                      ["両方が登録済み", both], ["一致0点", zero]]) {
        if (!example) continue;
        const [a, b] = example;
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.dataset.pair = pairKey(a, b);
        button.addEventListener("click", () => compare(a, b));
        byId("examples").append(button);
      }
      sides[0].button.addEventListener("click", () => openPhotoPicker({
        title: "左の写真を選ぶ", choices: photos, selected: sides[0].photo.id,
        onSelect: left => compare(left, sides[1].photo.id),
      }));
      sides[1].button.addEventListener("click", () => {
        const left = sides[0].photo.id;
        openPhotoPicker({
          title: "右の写真を選ぶ", choices: rightPhotos(left), selected: sides[1].photo.id,
          describe: photo => matchDetail(photo, left),
          onSelect: right => compare(left, right),
        });
      });
      const [a, b] = mixed || usable[0] || [photos[0].id, photos[1].id];
      graph = mountMatchGraph(byId("graph-view"), index, {
        fileLabel, registration, photoThumbnail, renderPhotoButton, openPhotoPicker,
        openPair(left, right) {
          switchView(false);
          compare(left, right);
        },
      });
      byId("show-graph").addEventListener("click", () => switchView(true));
      byId("show-comparison").addEventListener("click", () => switchView(false));
      byId("view-switch").hidden = false;
      switchView(true);
      await compare(a, b);
    }
    start().catch(error => {
      byId("summary").textContent = "";
      byId("error").textContent = error.message;
      byId("error").hidden = false;
    });
  </script>
</body>
</html>
'''


def gravity_vector(orientation):
    """World down expressed in camera coordinates (right, down, forward)."""
    if not isinstance(orientation, dict):
        return None
    try:
        b, g = (math.radians(float(orientation[k])) for k in ("beta_deg", "gamma_deg"))
        screen = float(orientation.get("screen_angle_deg", 0) or 0)
        if not all(math.isfinite(v) for v in (b, g, screen)):
            return None
        x, y = math.cos(b)*math.sin(g), math.sin(b)
        z = math.cos(b)*math.cos(g)
        a = math.radians(screen)
        return np.array([x*math.cos(a)-y*math.sin(a), x*math.sin(a)+y*math.cos(a), z])
    except (KeyError, TypeError, ValueError):
        return None


def vector_roll(vector):
    if vector is None or math.hypot(*vector[:2]) < .12:
        return None
    return (math.degrees(math.atan2(vector[0], vector[1])) + 180) % 360 - 180


def gravity_roll(orientation):
    return vector_roll(gravity_vector(orientation))


@lru_cache(maxsize=128)
def read_frames(path):
    result = {}
    try:
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if isinstance(row, dict) and isinstance(row.get("file"), str):
                        result[row["file"].replace("\\", "/")] = row
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return result


@lru_cache(maxsize=4096)
def file_digest(path, mtime, size):
    with path.open("rb") as stream:
        h = sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
        return h.hexdigest()


def digest_matches(path, expected):
    try:
        st = path.stat()
        return path.is_file() and file_digest(path, st.st_mtime_ns, st.st_size) == expected
    except OSError:
        return False


def capture_catalog():
    catalog = {}
    for meta in sorted((ROOT / "wifi-captures").glob("*/frames.jsonl")):
        for filename, row in read_frames(meta).items():
            candidate = meta.parent / filename
            if candidate.is_file():
                catalog.setdefault(candidate.name.casefold(), []).append((candidate, row, meta))
    return catalog


def locate_photo(row, recorded_paths, run, catalog):
    relative = Path(row["name"])
    expected = row["sha256"]
    capture = row.get("capture") or {}
    embedded = capture.get("orientation") or row.get("orientation") or (capture if "beta_deg" in capture and "gamma_deg" in capture else None)
    candidates = list(recorded_paths) + [ROOT / "input" / relative, ROOT / relative, run / relative, run / "images" / relative]
    existing = [p for p in candidates if p.is_file()]
    # Prefer the exact recorded image. Verify content before using adjacent sensor data.
    path = next((p for p in existing if digest_matches(p, expected)), None)
    if path is not None:
        for parent in (path.parent, *list(path.parents)[1:3]):
            meta = parent / "frames.jsonl"
            if not meta.is_file():
                continue
            try: key = path.relative_to(parent).as_posix()
            except ValueError: continue
            frame = read_frames(meta).get(key)
            if frame is not None:
                return path, embedded or frame.get("orientation"), "calibration.json" if embedded else str(meta)
    if embedded and path is not None:
        return path, embedded, "calibration.json"
    # Search capture sessions by filename AND SHA-256, not by sequence/filename alone.
    matches = [(p, frame, meta) for p, frame, meta in catalog.get(relative.name.casefold(), ()) if digest_matches(p, expected)]
    if matches:
        chosen, frame, meta = matches[0]
        unique = {json.dumps(f.get("orientation"), sort_keys=True) for _, f, _ in matches}
        orientation = embedded or (frame.get("orientation") if len(unique) == 1 else None)
        return path or chosen, orientation, "calibration.json" if embedded else str(meta) if orientation else None
    return path or (existing[0] if existing else ROOT / "input" / relative), embedded, "calibration.json" if embedded else None


class MatchRun:
    def __init__(self, run):
        calibration = read_json(run / "calibration.json")
        sources = {
            row["relative"]: row for row in calibration["sources"] if row["kind"] == "photo"
        }
        if len(sources) < 2:
            raise ValueError("写真が2枚以上あるカメラ推定を選んでください。")
        self.database = (run / calibration["database"]).resolve().as_uri() + "?mode=ro&immutable=1"
        reconstruction = pycolmap.Reconstruction(run / calibration["model"])
        registered = {}
        rotations = {}
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.images[image_id]
            try:
                pose = image.cam_from_world
                pose = pose() if callable(pose) else pose
                rotations[image.name] = np.asarray(pose.rotation.matrix(), dtype=float)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
            registered[image.name] = {
                index: point.point3D_id for index, point in enumerate(image.points2D) if point.has_point3D()
            }
        try:
            inputs = [Path(line) for line in (run / "inputs.txt").read_text(encoding="utf-8-sig").splitlines()]
        except FileNotFoundError:
            inputs = []
        by_filename = {}
        for path in inputs:
            if path.is_absolute():
                by_filename.setdefault(path.name.casefold(), []).append(path)

        photos, pairs = [], []
        catalog = capture_catalog()
        self.originals, self.tracks = {}, {}
        measured = []
        photo_names = {}
        with closing(self.connect()) as database:
            for image_id, name, width, height, features in database.execute(
                "SELECT i.image_id, i.name, c.width, c.height, COALESCE(k.rows, 0) FROM images i "
                "JOIN cameras c USING (camera_id) LEFT JOIN keypoints k USING (image_id) ORDER BY i.name"
            ):
                if name not in sources:
                    continue
                row = sources[name]
                relative = Path(row["name"])
                recorded_paths = (
                    path for path in by_filename.get(relative.name.casefold(), ())
                    if Path(*path.parts[-len(relative.parts):]) == relative
                )
                path, orientation, source = locate_photo(row, recorded_paths, run, catalog)
                vector = gravity_vector(orientation)
                roll = vector_roll(vector)
                photo_names[image_id] = (name, vector)
                if vector is not None and name in rotations:
                    measured.append(rotations[name].T @ vector)
                self.originals[image_id] = (path, row["sha256"], (width, height))
                if name in registered:
                    self.tracks[image_id] = registered[name]
                photos.append({
                    "id": image_id, "file": row["name"], "sequence": (row.get("capture") or {}).get("sequence"),
                    "registered": name in registered, "features": features, "available": digest_matches(path, row["sha256"]),
                    "roll": roll, "orientation_source": source,
                    "width": width, "height": height,
                })
            for pair_id, count in database.execute("SELECT pair_id, rows FROM two_view_geometries"):
                left, right = divmod(pair_id, PAIR_ID_LIMIT)
                if left in self.originals and right in self.originals:
                    pairs.append((left, right, count))
        if len(photos) < 2:
            raise ValueError("照合データベースに写真が2枚以上必要です。")
        # Only transfer gravity into the COLMAP world if multiple sensor readings agree.
        if len(measured) >= 3:
            v = np.asarray(measured)
            # Limit candidate count for large runs; all samples vote.
            candidates = v[np.linspace(0, len(v)-1, min(len(v),128), dtype=int)]
            votes = v @ candidates.T > math.cos(math.radians(15))
            mask = votes[:, int(np.argmax(votes.sum(axis=0)))]
            if mask.sum() >= max(3, math.ceil(len(v)*.60)):
                down = v[mask].mean(axis=0)
                down /= np.linalg.norm(down)
                for photo in photos:
                    name, vector = photo_names[photo["id"]]
                    if vector is None and name in rotations:
                        photo["roll"] = vector_roll(rotations[name] @ down)
                        if photo["roll"] is not None:
                            photo["orientation_source"] = "COLMAP + 撮影時の重力方向"
        self.photo_rolls = {p["id"]: p["roll"] for p in photos}
        self.index = {"run": run.name, "photos": photos, "pairs": pairs}

    def connect(self):
        return sqlite3.connect(self.database, uri=True)

    def photo(self, image_id, max_edge=None):
        path, digest, size = self.originals[image_id]
        original = path.read_bytes()
        if sha256(original).hexdigest() != digest:
            raise ValueError("保存時の元写真と内容が異なります。")
        output = BytesIO()
        with Image.open(BytesIO(original)) as image:
            photo = image if path.suffix.lower() == ".png" else ImageOps.exif_transpose(image)
            if photo.size != size:
                raise ValueError("元写真と照合データの解像度が異なります。")
            if max_edge is not None:
                photo.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            photo.convert("RGB").save(output, "JPEG", quality=82 if max_edge is not None else 92)
        return output.getvalue()

    @lru_cache(maxsize=128)
    def thumbnail(self, image_id, mode="off"):
        data = self.photo(image_id, max_edge=224)
        if mode == "off":
            return data
        roll = self.photo_rolls.get(image_id)
        if roll is None:
            return data
        angle = math.floor(roll / 90 + .5) * 90 if mode == "90" else roll
        output = BytesIO()
        with Image.open(BytesIO(data)) as im:
            im.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(35,59,62)).save(output, "JPEG", quality=88)
        return output.getvalue()

    def matches(self, left, right):
        if left == right or left not in self.originals or right not in self.originals:
            raise KeyError((left, right))
        first, second = sorted((left, right))
        with closing(self.connect()) as database:
            pair = database.execute(
                "SELECT rows, data FROM two_view_geometries WHERE pair_id = ?",
                (first * PAIR_ID_LIMIT + second,),
            ).fetchone()
            if pair is None or pair[0] == 0:
                return {"tested": pair is not None, "points": []}
            keypoints = dict(
                (image_id, (cols, data))
                for image_id, cols, data in database.execute(
                    "SELECT image_id, cols, data FROM keypoints WHERE image_id IN (?, ?)", (first, second)
                )
            )
        first_cols, first_data = keypoints[first]
        second_cols, second_data = keypoints[second]
        first_tracks, second_tracks = self.tracks.get(first), self.tracks.get(second)
        has_both_tracks = first_tracks is not None and second_tracks is not None
        structure_matches = 0
        points = []
        for a, b in struct.iter_unpack("<II", pair[1]):
            a_xy = struct.unpack_from("<ff", first_data, a * first_cols * 4)
            b_xy = struct.unpack_from("<ff", second_data, b * second_cols * 4)
            xy = a_xy + b_xy if left == first else b_xy + a_xy
            points.append([round(value, 2) for value in xy])
            a_track = first_tracks.get(a) if first_tracks is not None else None
            b_track = second_tracks.get(b) if second_tracks is not None else None
            if has_both_tracks:
                structure_matches += a_track is not None and a_track == b_track
            else:
                structure_matches += a_track is not None or b_track is not None
        return {"tested": True, "points": points, "structure_matches": structure_matches}


    @lru_cache(maxsize=32)
    def alignment(self, left, right):
        return estimate(self.matches(left, right)["points"], self.originals[left][2], self.originals[right][2])


class MatchHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_content(self, content_type, body, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, data, status=200):
        self.send_content("application/json; charset=utf-8", json.dumps(data, ensure_ascii=False).encode("utf-8"), status)

    def do_GET(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(400)
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            run = self.server.match_run
            if path.startswith("/run/"):
                parts = path.split("/", 3)
                name = unquote(parts[2])
                if len(parts) != 4 or name not in list_runs():
                    self.send_error(404)
                    return
                if parts[3] == "":
                    self.send_content("text/html; charset=utf-8", self.server.page)
                    return
                if parts[3] == "api/runs":
                    self.send_json({"runs": list_runs(), "selected": name})
                    return
                with self.server.run_lock:
                    run = self.server.run_cache.get(name)
                    if run is None:
                        run = select_run(name)
                        self.server.run_cache[name] = run
                        while len(self.server.run_cache) > 3:
                            self.server.run_cache.pop(next(iter(self.server.run_cache)))
                path = "/" + parts[3]
            parts = path.strip("/").split("/")
            if path == "/":
                self.send_content("text/html; charset=utf-8", self.server.page)
            elif path == "/match-graph.js":
                self.send_error(404)
            elif len(parts) == 4 and parts[:2] == ["api", "alignment"] and all(part.isdigit() for part in parts[2:]):
                self.send_json({"alignment": run.alignment(*map(int, parts[2:]))})
            elif path == "/api/runs":
                self.send_json({"runs": list_runs(), "selected": run.index["run"]})
            elif path == "/api/index":
                self.send_json(run.index)
            elif len(parts) == 3 and parts[:2] == ["api", "photo"] and parts[2].isdigit():
                self.send_content("image/jpeg", run.photo(int(parts[2])))
            elif len(parts) == 3 and parts[:2] == ["api", "thumb"] and parts[2].isdigit():
                mode = parse_qs(parsed.query).get("rotation", ["off"])[0]
                if mode not in ("off", "90", "free"):
                    self.send_error(400)
                    return
                self.send_content("image/jpeg", run.thumbnail(int(parts[2]), mode))
            elif len(parts) == 4 and parts[:2] == ["api", "pair"] and all(part.isdigit() for part in parts[2:]):
                self.send_json(run.matches(*map(int, parts[2:])))
            else:
                self.send_error(404)
        except (KeyError, FileNotFoundError):
            self.send_json({"error": "元写真か照合データが見つかりません。"}, 404)
        except (OSError, ValueError, RuntimeError, sqlite3.Error, struct.error) as error:
            self.send_json({"error": str(error)}, 422)

    def log_message(self, format, *args):
        pass


def list_runs():
    return [p.parent.name for p in sorted((ROOT / "runs").glob("*/calibration.json"), reverse=True)
            if RUN_NAME.fullmatch(p.parent.name)]


def select_run(name):
    runs = ROOT / "runs"
    if name:
        if not RUN_NAME.fullmatch(name):
            raise ValueError("実行フォルダ名を指定してください。")
        return MatchRun(runs / name)
    for calibration in sorted(runs.glob("*/calibration.json"), reverse=True):
        try:
            match_run = MatchRun(calibration.parent)
            if sum(photo["available"] for photo in match_run.index["photos"]) >= 2:
                return match_run
        except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error):
            continue
    raise ValueError("元写真と照合データの揃った実行が見つかりません。")


def main():
    parser = argparse.ArgumentParser(description="保存済みの写真とCOLMAPの対応点を表示します。")
    parser.add_argument("--run", help="照合を保存した実行フォルダ名。省略時は最新の写真の実行。")
    parser.add_argument("--port", type=int, default=0, help="指定したローカルポートで開きます。")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    register_heif_opener()
    match_run = select_run(args.run)
    with ThreadingHTTPServer(("127.0.0.1", args.port), MatchHandler) as server:
        server.match_run = match_run
        server.run_cache = {match_run.index["run"]: match_run}
        server.run_lock = RLock()
        server.page = APP_HTML.encode("utf-8")
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Feature matches: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, KeyError, sqlite3.Error) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
