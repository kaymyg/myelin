import json, re, struct, sys, os
import numpy as np
from atl import A, mesh, BY

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- name index
def split(name):
    n = name.lower()
    side = 'L' if re.search(r'\bleft\b', n) else 'R' if re.search(r'\bright\b', n) else ''
    base = re.sub(r'\s+', ' ', re.sub(r'\b(left|right)\b', '', n)).strip()
    return side, base

IDX = {}
for p in A['parts']:
    IDX.setdefault(split(p['name']), []).append(p)

def Vx(side, base):
    ps = IDX[(side, base.lower())]
    return np.vstack([mesh(p)[0] for p in ps])

def bbx(side, base):
    v = Vx(side, base); return v.min(0), v.max(0)

def cen(side, base):
    lo, hi = bbx(side, base); return (lo + hi) / 2

def sgn(side): return -1.0 if side == 'R' else 1.0   # right side of the body is -x

def off(side, lat=0, up=0, ant=0):
    return np.array([sgn(side) * lat, up, ant])

_ax = {}
def along(side, base, t, lat=0, up=0, ant=0, orient='down'):
    """centroid of a bone slice at fraction t of its long axis (0 = proximal), plus offset"""
    key = (side, base, orient)
    if key not in _ax:
        v = Vx(side, base); m = v.mean(0)
        u, s, vt = np.linalg.svd(v - m, full_matrices=False)
        a = vt[0]
        if orient == 'down' and a[1] > 0: a = -a
        if orient == 'fwd' and a[2] < 0: a = -a
        pr = (v - m) @ a
        _ax[key] = (v, pr, pr.min(), pr.max())
    v, pr, lo, hi = _ax[key]
    x = lo + t * (hi - lo)
    w = 0.05 * (hi - lo)
    sel = np.abs(pr - x) < w
    if sel.sum() < 3: sel = np.abs(pr - x) < 3 * w
    return v[sel].mean(0) + off(side, lat, up, ant)

# ---------------------------------------------------------------- spline
def smooth(pts, step=0.006):
    pts = np.array(pts, float)
    # drop duplicates
    keep = [0] + [i for i in range(1, len(pts)) if np.linalg.norm(pts[i] - pts[i - 1]) > 1e-5]
    pts = pts[keep]
    if len(pts) < 3:
        n = max(2, int(np.linalg.norm(pts[-1] - pts[0]) / step) + 1)
        return np.linspace(pts[0], pts[-1], n)
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0; t1 = t0 + np.linalg.norm(p1 - p0) ** .5; t2 = t1 + np.linalg.norm(p2 - p1) ** .5; t3 = t2 + np.linalg.norm(p3 - p2) ** .5
        n = max(2, int(np.linalg.norm(p2 - p1) / step) + 1)
        for t in np.linspace(t1, t2, n, endpoint=False):
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    out.append(pts[-1])
    return np.array(out)

def plen(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

# ---------------------------------------------------------------- spine
ORD = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth']
CERV = ['atlas', 'axis'] + [f'{o} cervical vertebra' for o in ORD[2:7]]
THOR = [f'{o} thoracic vertebra' for o in ORD]
LUMB = [f'{o} lumbar vertebra' for o in ORD[:5]]
VERT = CERV + THOR + LUMB

def canal(name):
    v = Vx('', name)
    lo, hi = v.min(0), v.max(0)
    mid = v[(np.abs(v[:, 0]) < 0.004)]
    z = np.sort(mid[:, 2])
    # largest gap in z along the midline is the vertebral canal
    zf = z[(z > lo[2] + 0.25 * (hi[2] - lo[2]))]
    g = np.diff(zf)
    if len(g) and g.max() > 0.006:
        k = g.argmax(); zc = (zf[k] + zf[k + 1]) / 2; zb = zf[k + 1]
    else:
        zc = lo[2] + 0.55 * (hi[2] - lo[2]); zb = zc + 0.008
    body = v[v[:, 2] > zb - 0.002]
    return dict(zc=zc, ytop=body[:, 1].max(), ybot=body[:, 1].min(), ymid=(body[:, 1].max() + body[:, 1].min()) / 2, zfront=hi[2])

CAN = {n: canal(n) for n in VERT}
CAN['atlas']['zc'] = CAN['axis']['zc'] + 0.002
med_lo, med_hi = bbx('', 'medulla oblongata')
MED_BOT = np.array([0, med_lo[1], (med_lo[2] + med_hi[2]) / 2 - 0.004])
sac_lo, sac_hi = bbx('', 'sacrum')
SACV = Vx('', 'sacrum')

def sac_front(yf, x):
    """anterior surface of the sacrum at height fraction yf (0 top) near |x|"""
    y = sac_hi[1] - yf * (sac_hi[1] - sac_lo[1])
    sel = SACV[(np.abs(SACV[:, 1] - y) < 0.008) & (np.abs(np.abs(SACV[:, 0]) - x) < 0.008)]
    z = sel[:, 2].max() if len(sel) else sac_hi[2]
    return y, z

# canal centre line as function of y
_cy = [MED_BOT[1]] + [CAN[n]['ymid'] for n in VERT]
_cz = [MED_BOT[2]] + [CAN[n]['zc'] for n in VERT]
for yf in (0.15, 0.4, 0.7, 0.95):
    y, z = sac_front(yf, 0.0)
    _cy.append(y); _cz.append(z - 0.022)
_cy = np.array(_cy); _cz = np.array(_cz)
o = np.argsort(_cy); _cy, _cz = _cy[o], _cz[o]
def zc(y): return float(np.interp(y, _cy, _cz))
def canal_pt(y, x=0.0): return np.array([x, y, zc(y)])

CONUS_Y = CAN['first lumbar vertebra']['ybot']
# spinal segments: name, cord level y, exit point
SEG = {}
def lin(a, b, n): return list(np.linspace(a, b, n))
cy = lin(CAN['atlas']['ytop'] + 0.004, CAN[CERV[6]]['ymid'], 8) + \
     lin(CAN[THOR[0]]['ytop'], CAN[THOR[9]]['ymid'], 12) + \
     lin(CAN[THOR[9]]['ymid'] - 0.008, CAN[THOR[11]]['ybot'] + 0.004, 5) + \
     lin(CAN[THOR[11]]['ybot'] - 0.002, CONUS_Y + 0.004, 6)
SEGN = [f'C{i}' for i in range(1, 9)] + [f'T{i}' for i in range(1, 13)] + [f'L{i}' for i in range(1, 6)] + [f'S{i}' for i in range(1, 6)] + ['Co1']
def exit_y(seg):
    k, n = seg[0], int(seg[1:]) if seg[0] != 'C' or seg[1] != 'o' else 1
    if seg == 'C1': return CAN['atlas']['ytop'] + 0.003, 0.014
    if seg[0] == 'C' and seg != 'Co1':
        if n <= 7: return (CAN[CERV[n - 2]]['ybot'] + CAN[CERV[n - 1]]['ytop']) / 2, 0.013
        return (CAN[CERV[6]]['ybot'] + CAN[THOR[0]]['ytop']) / 2, 0.014
    if seg[0] == 'T':
        nxt = THOR[n] if n < 12 else LUMB[0]
        return (CAN[THOR[n - 1]]['ybot'] + CAN[nxt]['ytop']) / 2, 0.016
    if seg[0] == 'L':
        if n < 5: return (CAN[LUMB[n - 1]]['ybot'] + CAN[LUMB[n]]['ytop']) / 2, 0.019
        return CAN[LUMB[4]]['ybot'] - 0.004, 0.021
    return None, None

for i, s in enumerate(SEGN):
    SEG[s] = dict(cord_y=cy[i], idx=i)

def foramen(side, seg):
    """point where the spinal nerve leaves the vertebral column"""
    if seg[0] == 'S' or seg == 'Co1':
        n = 5 if seg == 'Co1' else int(seg[1:]) - 1
        yf = [0.2, 0.4, 0.58, 0.74, 0.9, 0.99][n]
        x = [0.024, 0.022, 0.019, 0.016, 0.008, 0.004][n]
        y, z = sac_front(yf, x)
        return np.array([sgn(side) * x, y, z + 0.002])
    y, x = exit_y(seg)
    return np.array([sgn(side) * x, y, zc(y) + 0.008])

def root_line(side, seg):
    """rootlets from the cord segment down the canal to the foramen"""
    s = sgn(side); y0 = SEG[seg]['cord_y']; f = foramen(side, seg)
    pts = [canal_pt(y0, s * 0.0045)]
    if seg[0] == 'S' or seg == 'Co1':
        ys = np.arange(y0 - 0.02, sac_hi[1] - 0.01, -0.03)
        for y in ys: pts.append(canal_pt(y, s * 0.005))
        pts.append(np.array([s * abs(f[0]) * 0.8, f[1] + 0.012, f[2] - 0.018]))
    else:
        ys = np.arange(y0 - 0.015, f[1] + 0.012, -0.03)
        for y in ys: pts.append(canal_pt(y, s * 0.006))
        pts.append(canal_pt(f[1] + 0.004, s * 0.008))
    pts.append(f)
    return pts

def ramus(side, seg, d=0.016):
    f = foramen(side, seg)
    return f + off(side, d, -0.004, 0.010)

# ---------------------------------------------------------------- nerve builder
NERVES = []
class Nerve:
    def __init__(self, id, name, group, side, roots, kind, desc, vel=60):
        self.d = dict(id=id, name=name, group=group, side=side, roots=roots, kind=kind, desc=desc, vel=vel, lines=[])
        self.dense = []
        NERVES.append(self)
    def line(self, pts, r=0.0016, parent=None, at_end=True, delay=None):
        p = smooth(pts)
        d0 = 0.0
        if parent is not None:
            pp = self.dense[parent]; pd = self.d['lines'][parent]['d']
            if at_end:
                k = len(pp) - 1
            else:
                k = int(np.linalg.norm(pp - p[0], axis=1).argmin())
            p = np.vstack([pp[k], p[1:]]) if np.linalg.norm(pp[k] - p[0]) < 0.03 else np.vstack([pp[k], p])
            d0 = pd + plen(pp[:k + 1])
        if delay is not None: d0 = delay
        self.dense.append(p)
        self.d['lines'].append(dict(p=p, r=r, d=d0, par=parent, k=(None if parent is None else k)))
        return len(self.dense) - 1
    def route(self, i):
        """dense points from the origin of the nerve to the end of line i"""
        chain = []
        while i is not None:
            L = self.d['lines'][i]; chain.append((i, L)); i = L['par']
        chain.reverse()
        out = []
        for j, (i, L) in enumerate(chain):
            p = self.dense[i]
            if j + 1 < len(chain):
                p = p[:chain[j + 1][1]['k'] + 1]
            out.append(p if not out else p[1:])
        return np.vstack(out)

def plexus(n, side, segs, conv, r=0.0012):
    """root contributions from several segments converging on one point; returns index of the trunk seed"""
    ids = []
    L = 0
    for s in segs:
        i = n.line([foramen(side, s), ramus(side, s), conv], r)
        ids.append(i); L = max(L, plen(n.dense[i]))
    return ids, L

SIDES = [('R', 'Right'), ('L', 'Left')]
NV = {}

# ---------------------------------------------------------------- spinal cord
cord = Nerve('cord', 'Spinal cord', 'cns', '', 'C1 to Co1', 'mixed',
             'The cable of the central nervous system. It runs inside the vertebral canal from the medulla to about the L1/L2 vertebra, carries motor commands down and sensation up, and runs reflexes on its own. Below its tip the lumbar and sacral roots continue as the cauda equina.', 70)
ys = np.arange(MED_BOT[1], CONUS_Y, -0.02)
cord.line([canal_pt(y) for y in ys] + [canal_pt(CONUS_Y)], 0.0055)
cord.line([canal_pt(CONUS_Y), canal_pt(CONUS_Y - 0.06), canal_pt(sac_hi[1] - 0.02), canal_pt(sac_lo[1] + 0.03)], 0.0009, parent=0)
NV['cord'] = cord

REGION = dict(C='cervical', T='thoracic', L='lumbar', S='sacral')
for S, SN in SIDES:
    for s in SEGN:
        reg = 'coccygeal' if s == 'Co1' else REGION[s[0]]
        n = Nerve(f'root_{s}_{S}', f'{SN} {s} spinal nerve', 'roots', S, s, 'mixed',
                  f'One of the 31 pairs of spinal nerves ({reg}). A dorsal root brings sensation in, a ventral root carries motor commands out, and they join where the nerve leaves the spine.', 60)
        n.line(root_line(S, s), 0.0011)
        n.line([foramen(S, s), ramus(S, s, 0.012)], 0.0012, parent=0)
        NV[n.d['id']] = n

# ---------------------------------------------------------------- limbs, trunk
def build_side(S, SN):
    s = sgn(S)
    P = lambda base, t, **k: along(S, base, t, **k)
    C = lambda base: cen(S, base)
    O = lambda **k: off(S, **k)
    def mk(id, name, group, roots, kind, desc, vel=60):
        n = Nerve(f'{id}_{S}', f'{SN} {name}', group, S, roots, kind, desc, vel); NV[n.d['id']] = n; return n

    # ---------------- cervical plexus
    c4 = ramus(S, 'C4')
    heart_lo, heart_hi = bbx('', 'wall of ventricle')
    dia_lo, dia_hi = bbx('', 'diaphragm')
    n = mk('phrenic', 'phrenic nerve', 'cervical', 'C3 to C5', 'mixed',
           'The only motor supply to the diaphragm, so it drives every breath. Runs from the neck down through the chest beside the heart. "C3, 4, 5 keeps the diaphragm alive."', 60)
    conv = c4 + O(lat=0.012, up=-0.02, ant=0.012)
    ids, L = plexus(n, S, ['C3', 'C4', 'C5'], conv)
    hx = (heart_hi[0] + 0.012) if S == 'L' else (heart_lo[0] - 0.02)
    n.line([conv, np.array([s * 0.035, CAN[THOR[0]]['ymid'], zc(CAN[THOR[0]]['ymid']) + 0.05]),
            np.array([hx if S == 'L' else min(hx, -0.04), heart_hi[1] + 0.02, 0.03]),
            np.array([hx if S == 'L' else min(hx, -0.045), (heart_lo[1] + heart_hi[1]) / 2, 0.035]),
            np.array([s * 0.055, dia_hi[1] - 0.012, 0.03])], 0.0014, delay=L)

    n = mk('great_auricular', 'great auricular nerve', 'cervical', 'C2, C3', 'sensory',
           'Skin sensation over the ear, the angle of the jaw and the parotid region.', 50)
    conv = ramus(S, 'C3') + O(lat=0.02, ant=-0.004)
    ids, L = plexus(n, S, ['C2', 'C3'], conv)
    ear = np.array([s * 0.078, 1.578, -0.012])
    n.line([conv, conv + O(lat=0.018, up=0.03, ant=0.004), ear + O(lat=-0.004, up=-0.025), ear], 0.001, delay=L)

    n = mk('lesser_occipital', 'lesser occipital nerve', 'cervical', 'C2', 'sensory',
           'Skin sensation of the scalp behind and above the ear.', 50)
    f = ramus(S, 'C2')
    n.line([foramen(S, 'C2'), f, f + O(lat=0.03, up=0.02, ant=-0.02), np.array([s * 0.068, 1.615, -0.055]), np.array([s * 0.066, 1.65, -0.06])], 0.001)

    n = mk('greater_occipital', 'greater occipital nerve', 'cervical', 'C2 (dorsal ramus)', 'sensory',
           'Sensation of the back of the scalp up to the top of the head. Irritation here causes occipital neuralgia.', 50)
    f = foramen(S, 'C2')
    n.line([f, f + O(lat=0.008, ant=-0.03), np.array([s * 0.03, 1.575, -0.105]), np.array([s * 0.035, 1.63, -0.118]), np.array([s * 0.035, 1.68, -0.095])], 0.0011)

    n = mk('supraclavicular', 'supraclavicular nerves', 'cervical', 'C3, C4', 'sensory',
           'Skin sensation over the collarbone, the top of the shoulder and the upper chest.', 50)
    conv = c4 + O(lat=0.022, up=-0.01)
    ids, L = plexus(n, S, ['C3', 'C4'], conv)
    m = n.line([conv, conv + O(lat=0.03, up=-0.045, ant=0.02)], 0.001, delay=L)
    cl = lambda t, **k: along(S, 'clavicle', t, orient='any', **k)
    clv = Vx(S, 'clavicle'); xs = np.abs(clv[:, 0])
    def clav(t):
        x = xs.min() + t * (xs.max() - xs.min()); sel = clv[np.abs(xs - x) < 0.008]; return sel.mean(0)
    for t in (0.2, 0.55, 0.9):
        e = clav(t)
        n.line([conv + O(lat=0.03, up=-0.045, ant=0.02), e + np.array([0, 0.012, 0.012]), e + np.array([0, -0.03, 0.02])], 0.0008, parent=m)

    # ---------------- brachial plexus
    hum = 'humerus'
    cords = clav(0.5) + np.array([0, -0.022, -0.012])          # behind the mid clavicle, over the first rib
    axilla = P(hum, 0.14, lat=-0.034, ant=0.004)
    def bp(n, segs):
        ids, L = plexus(n, S, segs, cords, 0.0014)
        return L
    wrist_c = (C('capitate') + C('lunate')) / 2
    def finger(name, ph='distal', **k):
        return cen(S, f'{ph} phalanx of {name}') + O(**k)

    n = mk('median', 'median nerve', 'brachial', 'C5 to T1', 'mixed',
           'Runs down the middle of the arm and through the carpal tunnel. Bends the wrist and fingers, moves the thumb, and feels the palm side of the thumb, index, middle and half of the ring finger. Squeezed in carpal tunnel syndrome.', 60)
    L = bp(n, ['C5', 'C6', 'C7', 'C8', 'T1'])
    m = n.line([cords, axilla, P(hum, 0.35, lat=-0.022, ant=0.010), P(hum, 0.65, lat=-0.02, ant=0.014), P(hum, 0.9, lat=-0.012, ant=0.022),
                (P('radius', 0.12) + P('ulna', 0.12)) / 2 + O(ant=0.022), (P('radius', 0.5) + P('ulna', 0.5)) / 2 + O(ant=0.016),
                (P('radius', 0.9) + P('ulna', 0.9)) / 2 + O(ant=0.014), wrist_c + O(ant=0.012), C('third metacarpal bone') + O(ant=0.012, up=0.012)], 0.0021, delay=L)
    for fn, lt in (('thumb', 0.0), ('index finger', 0.003), ('middle finger', 0.0), ('ring finger', 0.004)):
        mc = {'thumb': 'first', 'index finger': 'second', 'middle finger': 'third', 'ring finger': 'fourth'}[fn]
        n.line([C('third metacarpal bone') + O(ant=0.012, up=0.012), C(f'{mc} metacarpal bone') + O(ant=0.010, lat=lt),
                finger(fn, 'proximal', ant=0.008, lat=lt), finger(fn, ant=0.006, lat=lt)], 0.0009, parent=m)

    n = mk('ulnar', 'ulnar nerve', 'brachial', 'C8, T1', 'mixed',
           'Passes behind the inner elbow, the "funny bone", then down the little finger side of the forearm. Works most small hand muscles and feels the little finger and half of the ring finger.', 60)
    L = bp(n, ['C8', 'T1'])
    m = n.line([cords, axilla + O(up=-0.004, ant=-0.006), P(hum, 0.4, lat=-0.026, ant=-0.002), P(hum, 0.75, lat=-0.028, ant=-0.01), P(hum, 0.965, lat=-0.034, ant=-0.016),
                P('ulna', 0.2, lat=-0.012, ant=0.012), P('ulna', 0.55, lat=-0.01, ant=0.014), P('ulna', 0.92, lat=-0.008, ant=0.014),
                C('pisiform') + O(lat=0.004, ant=0.004), C('fifth metacarpal bone') + O(ant=0.010, up=0.01)], 0.0019, delay=L)
    n.line([C('fifth metacarpal bone') + O(ant=0.010, up=0.01), finger('little finger', 'proximal', ant=0.008), finger('little finger', ant=0.006)], 0.0009, parent=m)
    n.line([C('fifth metacarpal bone') + O(ant=0.010, up=0.01), C('fourth metacarpal bone') + O(ant=0.010, lat=-0.004), finger('ring finger', 'proximal', ant=0.008, lat=-0.004), finger('ring finger', ant=0.006, lat=-0.004)], 0.0009, parent=m)

    n = mk('radial', 'radial nerve', 'brachial', 'C5 to T1', 'mixed',
           'Spirals around the back of the humerus. Straightens the elbow, wrist and fingers and feels the back of the hand on the thumb side. Injury causes wrist drop.', 60)
    L = bp(n, ['C5', 'C6', 'C7', 'C8', 'T1'])
    elbow = P(hum, 0.94, lat=0.026, ant=0.012)
    m = n.line([cords, axilla + O(ant=-0.012, up=-0.006), P(hum, 0.3, lat=-0.016, ant=-0.02), P(hum, 0.48, lat=0.0, ant=-0.02), P(hum, 0.66, lat=0.018, ant=-0.008),
                P(hum, 0.82, lat=0.022, ant=0.006), elbow], 0.0021, delay=L)
    b = n.line([elbow, P('radius', 0.25, lat=0.012, ant=0.012), P('radius', 0.6, lat=0.014, ant=0.006), P('radius', 0.92, lat=0.012, ant=-0.006),
                C('first metacarpal bone') + O(ant=-0.01, lat=0.004), finger('thumb', ant=-0.006)], 0.0013, parent=m)
    n.line([P('radius', 0.92, lat=0.012, ant=-0.006), C('second metacarpal bone') + O(ant=-0.01), finger('index finger', 'proximal', ant=-0.008)], 0.0009, parent=b, at_end=False)
    n.line([elbow, P('radius', 0.18, lat=0.004, ant=-0.012), (P('radius', 0.5) + P('ulna', 0.5)) / 2 + O(ant=-0.016), (P('radius', 0.9) + P('ulna', 0.9)) / 2 + O(ant=-0.012)], 0.0012, parent=m)

    n = mk('musculocutaneous', 'musculocutaneous nerve', 'brachial', 'C5 to C7', 'mixed',
           'Powers the biceps and brachialis, which bend the elbow, then continues as a skin nerve to the thumb side of the forearm.', 60)
    L = bp(n, ['C5', 'C6', 'C7'])
    n.line([cords, axilla + O(lat=0.012, ant=0.008), P(hum, 0.3, lat=-0.004, ant=0.02), P(hum, 0.6, lat=0.004, ant=0.024), P(hum, 0.9, lat=0.016, ant=0.024),
            P('radius', 0.3, lat=0.012, ant=0.016), P('radius', 0.7, lat=0.012, ant=0.012)], 0.0015, delay=L)

    n = mk('axillary', 'axillary nerve', 'brachial', 'C5, C6', 'mixed',
           'Wraps around the neck of the humerus to the deltoid, which lifts the arm sideways, and feels the "regimental badge" patch of skin on the shoulder.', 60)
    L = bp(n, ['C5', 'C6'])
    n.line([cords, axilla + O(ant=-0.008, up=0.012), P(hum, 0.16, lat=-0.012, ant=-0.022), P(hum, 0.17, lat=0.012, ant=-0.02), P(hum, 0.18, lat=0.019, ant=0.0), P(hum, 0.2, lat=0.014, ant=0.018)], 0.0015, delay=L)

    n = mk('long_thoracic', 'long thoracic nerve', 'brachial', 'C5 to C7', 'motor',
           'Runs down the side of the rib cage to serratus anterior, which holds the shoulder blade against the chest. Injury causes a winged scapula.', 60)
    L = bp(n, ['C5', 'C6', 'C7'])
    ribpts = []
    for o in ('third', 'fifth', 'seventh'):
        rv = Vx(S, f'{o} rib'); k = np.abs(rv[:, 0]).argmax(); ribpts.append(rv[k] + O(lat=0.006))
    n.line([cords, cords + O(lat=0.004, up=-0.03, ant=-0.01)] + ribpts, 0.0011, delay=L)

    # ---------------- intercostal nerves
    def rib_path(o, drop=0.007):
        rv = Vx(S, f'{o} rib')
        zc0 = (rv[:, 2].min() + rv[:, 2].max()) / 2
        th = np.arctan2(np.abs(rv[:, 0]), rv[:, 2] - zc0)       # 0 = front, pi = back
        pts = []
        for a in np.linspace(th.max() - 0.12, th.min() + 0.05, 9):
            sel = rv[np.abs(th - a) < 0.14]
            if len(sel) > 2: pts.append(sel.mean(0) + np.array([0, -drop, 0]))
        return pts
    for i in range(1, 12):
        seg = f'T{i}'
        n = mk(f'intercostal{i}', f'{ORD[i - 1]} intercostal nerve', 'intercostal', seg, 'mixed',
               'Runs in the groove under its rib. Works the intercostal muscles used in breathing and feels a band of skin around the chest' + (' and abdominal wall.' if i >= 7 else '.'), 55)
        rp = rib_path(ORD[i - 1])
        pts = [foramen(S, seg), ramus(S, seg)] + rp
        end = rp[-1]
        if i >= 7:
            ya = 1.20 - (i - 7) * 0.045
            pts += [np.array([s * 0.09, (end[1] + ya) / 2 - 0.01, 0.105]), np.array([s * 0.035, ya, 0.118]), np.array([s * 0.006, ya - 0.006, 0.12])]
        elif i <= 6:
            cc = IDX.get((S, f'{ORD[i - 1]} costal cartilage'))
            if cc:
                cv = Vx(S, f'{ORD[i - 1]} costal cartilage'); k = np.abs(cv[:, 0]).argmin()
                pts += [cv.mean(0) + np.array([0, -0.007, 0]), cv[k] + np.array([0, -0.007, 0.004])]
        n.line(pts, 0.001)
    n = mk('subcostal', 'subcostal nerve', 'intercostal', 'T12', 'mixed',
           'The twelfth thoracic nerve, running below the last rib to the lower abdominal wall.', 55)
    rp = rib_path('twelfth', 0.012)
    n.line([foramen(S, 'T12'), ramus(S, 'T12')] + rp + [np.array([s * 0.125, 1.02, 0.03]), np.array([s * 0.07, 0.975, 0.105]), np.array([s * 0.01, 0.955, 0.115])], 0.001)

    # ---------------- lumbar plexus
    hipv = Vx(S, 'hip bone'); hlo, hhi = hipv.min(0), hipv.max(0)
    top = hipv[hipv[:, 1] > hhi[1] - 0.06]
    asis = top[top[:, 2].argmax()]                                   # anterior superior iliac spine
    crest = hipv[hipv[:, 1].argmax()]
    pub = hipv[(np.abs(hipv[:, 0])).argmin()]                         # near the pubic symphysis
    low = hipv[hipv[:, 1] < hlo[1] + 0.02]
    isch = low.mean(0)                                               # ischial tuberosity
    fem = 'femur'
    ming = (asis + pub) / 2 + np.array([0, -0.004, 0.006])           # under the inguinal ligament
    psoas = lambda seg, d=0.03: foramen(S, seg) + O(lat=d, up=-0.02, ant=0.022)

    n = mk('femoral', 'femoral nerve', 'lumbar', 'L2 to L4', 'mixed',
           'The main nerve of the front of the thigh. Straightens the knee through the quadriceps and feels the front of the thigh. It carries both limbs of the knee jerk reflex.', 60)
    conv = psoas('L4', 0.034)
    ids, L = plexus(n, S, ['L2', 'L3', 'L4'], conv, 0.0014)
    m = n.line([conv, np.array([s * 0.075, (conv[1] + ming[1]) / 2, 0.03]), ming + O(lat=0.012), P(fem, 0.1, lat=-0.02, ant=0.045)], 0.0021, delay=L)
    q = n.line([P(fem, 0.1, lat=-0.02, ant=0.045), P(fem, 0.3, lat=0.0, ant=0.05), P(fem, 0.55, lat=0.006, ant=0.048), P(fem, 0.8, lat=0.004, ant=0.04), C('patella') + O(up=0.03, ant=0.004)], 0.0014, parent=m)
    n.line([P(fem, 0.1, lat=-0.02, ant=0.045), P(fem, 0.3, lat=0.03, ant=0.035), P(fem, 0.55, lat=0.04, ant=0.02)], 0.001, parent=m)
    NV[f'femoral_{S}'].quad = q

    n = mk('saphenous', 'saphenous nerve', 'lumbar', 'L3, L4', 'sensory',
           'The longest skin branch of the femoral nerve. Feels the inner side of the knee, leg, ankle and foot.', 50)
    ids, L = plexus(n, S, ['L3', 'L4'], conv, 0.0012)
    n.line([conv, np.array([s * 0.075, (conv[1] + ming[1]) / 2, 0.028]), ming + O(lat=0.008, ant=-0.002), P(fem, 0.25, lat=-0.03, ant=0.035), P(fem, 0.6, lat=-0.04, ant=0.012), P(fem, 0.95, lat=-0.045, ant=-0.01),
            P('tibia', 0.15, lat=-0.03, ant=0.0), P('tibia', 0.55, lat=-0.022, ant=0.006), P('tibia', 0.97, lat=-0.022, ant=0.008), C('navicular bone of foot') + O(lat=-0.016, up=0.004)], 0.0011, delay=L)

    n = mk('obturator', 'obturator nerve', 'lumbar', 'L2 to L4', 'mixed',
           'Leaves the pelvis through the obturator foramen to the inner thigh. Works the adductors, which pull the legs together.', 60)
    conv2 = psoas('L4', 0.022) + O(ant=-0.008)
    ids, L = plexus(n, S, ['L2', 'L3', 'L4'], conv2, 0.0012)
    obt = np.array([pub[0] + s * 0.035, pub[1] + 0.012, pub[2] - 0.022])
    n.line([conv2, np.array([s * 0.045, sac_hi[1] - 0.03, -0.005]), obt, P(fem, 0.25, lat=-0.05, ant=0.012), P(fem, 0.55, lat=-0.045, ant=0.006)], 0.0015, delay=L)

    n = mk('lat_fem_cut', 'lateral femoral cutaneous nerve', 'lumbar', 'L2, L3', 'sensory',
           'Pure skin nerve of the outer thigh. Trapped under the inguinal ligament it causes meralgia paraesthetica, a burning outer thigh.', 50)
    conv3 = psoas('L3', 0.04)
    ids, L = plexus(n, S, ['L2', 'L3'], conv3, 0.001)
    n.line([conv3, np.array([asis[0] - s * 0.03, (conv3[1] + asis[1]) / 2, 0.02]), asis + O(lat=-0.012, up=-0.01, ant=0.004), P(fem, 0.25, lat=0.05, ant=0.03), P(fem, 0.6, lat=0.05, ant=0.02)], 0.001, delay=L)

    n = mk('iliohypogastric', 'iliohypogastric nerve', 'lumbar', 'L1', 'mixed',
           'Runs around the flank above the hip bone to the lower abdominal wall and the skin above the pubis.', 55)
    f = foramen(S, 'L1')
    n.line([f, ramus(S, 'L1'), np.array([s * 0.1, crest[1] + 0.03, -0.03]), np.array([s * 0.135, crest[1] + 0.012, 0.03]), np.array([s * 0.09, asis[1] + 0.012, 0.095]), np.array([s * 0.025, pub[1] + 0.05, 0.1])], 0.001)
    n = mk('ilioinguinal', 'ilioinguinal nerve', 'lumbar', 'L1', 'mixed',
           'Follows the iliohypogastric nerve a little lower, then passes through the inguinal canal to the skin of the groin.', 55)
    n.line([f, ramus(S, 'L1') + O(up=-0.01), np.array([s * 0.1, crest[1] + 0.012, -0.03]), np.array([s * 0.13, crest[1] - 0.006, 0.03]), asis + O(lat=-0.014, up=-0.004, ant=0.012), ming + O(lat=-0.012, ant=0.014), pub + O(lat=0.016, up=0.006, ant=0.016)], 0.001)
    n = mk('genitofemoral', 'genitofemoral nerve', 'lumbar', 'L1, L2', 'mixed',
           'Runs down the front of the psoas muscle to the groin, supplying skin of the upper inner thigh and the genital region.', 55)
    conv4 = psoas('L2', 0.026) + O(ant=0.006)
    ids, L = plexus(n, S, ['L1', 'L2'], conv4, 0.001)
    n.line([conv4, np.array([s * 0.055, sac_hi[1] + 0.0, 0.035]), ming + O(lat=-0.004, up=0.008, ant=0.008), ming + O(lat=-0.006, up=-0.035, ant=0.016)], 0.001, delay=L)

    # ---------------- sacral plexus
    sx, s2y = 0.045, sac_front(0.5, 0.03)[0]
    notch = np.array([s * 0.062, s2y - 0.012, sac_front(0.5, 0.03)[1] + 0.0])     # greater sciatic foramen
    conv_s = np.array([s * 0.04, s2y + 0.008, sac_front(0.45, 0.03)[1] + 0.008])
    gt = P(fem, 0.08, lat=0.03)                                                    # greater trochanter side
    midbut = (isch + gt) / 2 + np.array([0, 0.0, -0.022])
    pop = P(fem, 0.78, ant=-0.03)

    def sp(n, segs, r=0.0014):
        return plexus(n, S, segs, conv_s, r)[1]

    n = mk('sciatic', 'sciatic nerve', 'sacral', 'L4 to S3', 'mixed',
           'The thickest and longest nerve in the body. Leaves the pelvis under the buttock and runs down the back of the thigh, working the hamstrings, then splits above the knee into the tibial and common fibular nerves. Irritation is felt as sciatica.', 60)
    L = sp(n, ['L4', 'L5', 'S1', 'S2', 'S3'], 0.0016)
    n.line([conv_s, notch, midbut, P(fem, 0.3, lat=-0.004, ant=-0.032), P(fem, 0.55, lat=-0.004, ant=-0.03), pop], 0.0034, delay=L)
    sci_route = [conv_s, notch, midbut, P(fem, 0.3, lat=-0.004, ant=-0.032), P(fem, 0.55, lat=-0.004, ant=-0.03), pop]

    tib = 'tibia'
    n = mk('tibial', 'tibial nerve', 'sacral', 'L4 to S3', 'mixed',
           'The larger branch of the sciatic nerve. Runs down the back of the calf, works the muscles that point the foot and curl the toes, then passes behind the inner ankle to the sole.', 60)
    L = sp(n, ['L4', 'L5', 'S1', 'S2', 'S3'])
    ank = P(tib, 0.975, lat=-0.02, ant=-0.022)
    m = n.line(sci_route + [P(fem, 0.97, ant=-0.034), P(tib, 0.15, lat=0.006, ant=-0.034), P(tib, 0.5, lat=0.0, ant=-0.03), P(tib, 0.85, lat=-0.012, ant=-0.026), ank,
                            C('calcaneus') + O(lat=-0.018, up=-0.012, ant=0.03)], 0.0022, delay=L)
    heel = C('calcaneus') + O(lat=-0.018, up=-0.012, ant=0.03)
    def toe(nm, ph='distal', **k):
        key = (S, f'{ph} phalanx of {nm}')
        if key not in IDX: key = (S, f'proximal phalanx of {nm}')
        return cen(*key) + O(**k)
    mp = n.line([heel, C('navicular bone of foot') + O(up=-0.022, lat=-0.004), C('first metatarsal bone') + O(up=-0.016), toe('big toe', up=-0.008)], 0.0012, parent=m)
    for nm, mt in (('second toe', 'second'), ('third toe', 'third')):
        n.line([C('navicular bone of foot') + O(up=-0.022, lat=-0.004), C(f'{mt} metatarsal bone') + O(up=-0.016), toe(nm, up=-0.008)], 0.0008, parent=mp, at_end=False)
    lp = n.line([heel, C('cuboid bone') + O(up=-0.02), C('fifth metatarsal bone') + O(up=-0.014), toe('little toe', up=-0.008)], 0.0011, parent=m)
    n.line([C('cuboid bone') + O(up=-0.02), C('fourth metatarsal bone') + O(up=-0.016), toe('fourth toe', up=-0.008)], 0.0008, parent=lp, at_end=False)
    n.mp = mp; n.main = m

    fib = 'fibula'
    n = mk('common_fibular', 'common fibular (peroneal) nerve', 'sacral', 'L4 to S2', 'mixed',
           'The smaller branch of the sciatic nerve. Winds around the neck of the fibula, where it is easily injured, then divides. The deep branch lifts the foot and toes, the superficial branch turns the foot outward and feels the top of the foot. Injury causes foot drop.', 60)
    L = sp(n, ['L4', 'L5', 'S1', 'S2'])
    neck = P(fib, 0.06, lat=0.007, ant=0.0)
    m = n.line(sci_route + [P(fem, 0.95, lat=0.03, ant=-0.03), P(fib, 0.02, lat=0.006, ant=-0.014), neck], 0.0018, delay=L)
    dorsum = C('intermediate cuneiform bone') + O(up=0.016)
    dp = n.line([neck, (P(tib, 0.2) + P(fib, 0.18)) / 2 + O(ant=0.022), (P(tib, 0.55) + P(fib, 0.55)) / 2 + O(ant=0.024), P(tib, 0.97, lat=0.006, ant=0.024), dorsum,
                 (C('first metatarsal bone') + C('second metatarsal bone')) / 2 + O(up=0.014), (toe('big toe', 'proximal') + toe('second toe', 'proximal')) / 2 + O(up=0.01)], 0.0012, parent=m)
    sf = n.line([neck, P(fib, 0.3, lat=0.014, ant=0.012), P(fib, 0.65, lat=0.012, ant=0.02), P(fib, 0.95, lat=0.004, ant=0.03), C('cuboid bone') + O(up=0.02, lat=-0.012), C('third metatarsal bone') + O(up=0.014), toe('third toe', 'proximal', up=0.01)], 0.0011, parent=m)
    n.line([C('cuboid bone') + O(up=0.02, lat=-0.012), C('fourth metatarsal bone') + O(up=0.014), toe('fourth toe', 'proximal', up=0.01)], 0.0008, parent=sf, at_end=False)
    n.dp = dp

    n = mk('sural', 'sural nerve', 'sacral', 'S1, S2', 'sensory',
           'Skin nerve of the back of the calf and the outer edge of the foot. Often used as a graft or biopsy nerve.', 50)
    L = sp(n, ['S1', 'S2'], 0.001)
    n.line(sci_route + [P(tib, 0.2, lat=0.008, ant=-0.045), P(tib, 0.6, lat=0.014, ant=-0.042), P(fib, 0.97, lat=0.008, ant=-0.02), C('calcaneus') + O(lat=0.024, up=0.0, ant=0.02), C('fifth metatarsal bone') + O(lat=0.012, up=0.004)], 0.001, delay=L)

    iliac_lat = hipv[np.abs(hipv[:, 0]).argmax()]
    n = mk('sup_gluteal', 'superior gluteal nerve', 'sacral', 'L4 to S1', 'motor',
           'Works gluteus medius and minimus, which keep the pelvis level when you stand on one leg.', 60)
    L = sp(n, ['L4', 'L5', 'S1'], 0.001)
    n.line([conv_s, notch + O(up=0.012), notch + O(lat=0.03, up=0.03, ant=-0.012), np.array([iliac_lat[0], iliac_lat[1] - 0.02, iliac_lat[2] - 0.02])], 0.0011, delay=L)
    n = mk('inf_gluteal', 'inferior gluteal nerve', 'sacral', 'L5 to S2', 'motor',
           'Works gluteus maximus, the big muscle that extends the hip when you climb stairs or stand up.', 60)
    L = sp(n, ['L5', 'S1', 'S2'], 0.001)
    n.line([conv_s, notch + O(up=-0.006), midbut + O(lat=-0.012, up=0.02, ant=-0.03), midbut + O(lat=0.0, up=-0.01, ant=-0.045)], 0.0011, delay=L)
    n = mk('post_fem_cut', 'posterior femoral cutaneous nerve', 'sacral', 'S1 to S3', 'sensory',
           'Skin sensation of the lower buttock and the back of the thigh down to the knee.', 50)
    L = sp(n, ['S1', 'S2', 'S3'], 0.001)
    n.line([conv_s, notch + O(lat=-0.006, up=-0.008), midbut + O(lat=-0.014, ant=-0.012), P(fem, 0.3, lat=-0.012, ant=-0.05), P(fem, 0.6, lat=-0.01, ant=-0.05), P(fem, 0.85, ant=-0.045)], 0.001, delay=L)
    n = mk('pudendal', 'pudendal nerve', 'sacral', 'S2 to S4', 'mixed',
           'The main nerve of the perineum. Controls the external sphincters of bladder and bowel and carries genital sensation.', 55)
    conv_p = np.array([s * 0.03, sac_front(0.72, 0.02)[0], sac_front(0.72, 0.02)[1] + 0.006])
    ids, L = plexus(n, S, ['S2', 'S3', 'S4'], conv_p, 0.001)
    n.line([conv_p, np.array([s * 0.05, isch[1] + 0.055, isch[2] - 0.0]), np.array([s * 0.04, isch[1] + 0.02, isch[2] + 0.02]), np.array([s * 0.018, pub[1] - 0.02, pub[2] - 0.02])], 0.0011, delay=L)

    # ---------------- sympathetic trunk
    n = mk('symp', 'sympathetic trunk', 'autonomic', 'T1 to L2 outflow', 'autonomic',
           'A chain of ganglia beside the spine. It carries the "fight or flight" outflow: faster heart, wider pupils and airways, sweating, blood sent to muscle.', 5)
    pts = []; gang = []
    for nm in VERT:
        cc = CAN[nm]; lat = 0.024 if 'cervical' in nm or nm in ('atlas', 'axis') else 0.026 if 'thoracic' in nm else 0.024
        p = np.array([s * lat, cc['ymid'], cc['zc'] + (0.022 if 'thoracic' not in nm else 0.016)])
        if 'lumbar' in nm: p[2] = cc['zc'] + 0.034
        pts.append(p)
        if nm in ('axis', CERV[5], CERV[6]) or 'thoracic' in nm or 'lumbar' in nm: gang.append(p)
    for yf, x in ((0.25, 0.018), (0.55, 0.014), (0.85, 0.008), (0.99, 0.002)):
        y, z = sac_front(yf, x); p = np.array([s * x, y, z + 0.004]); pts.append(p); gang.append(p)
    n.line(pts, 0.0011)
    n.d['ganglia'] = gang

for S, SN in SIDES: build_side(S, SN)

# ---------------------------------------------------------------- cranial nerves
mid_lo, mid_hi = bbx('', 'midbrain'); pons_lo, pons_hi = bbx('', 'pons')
MID = (mid_lo + mid_hi) / 2; PONS = (pons_lo + pons_hi) / 2; MED = (med_lo + med_hi) / 2
eth_lo, eth_hi = bbx('', 'ethmoid')
chi = cen('', 'optic chiasm')
heart_lo, heart_hi = bbx('', 'wall of ventricle'); HEART = (heart_lo + heart_hi) / 2
sto_lo, sto_hi = bbx('', 'stomach')
tong = cen('', 'tongue'); thy = cen('', 'thyroid cartilage')
manv = Vx('', 'mandible')

def cranial(S, SN):
    s = sgn(S); O = lambda **k: off(S, **k)
    eye = cen(S, 'sclera')
    apex = eye + O(lat=-0.012, ant=-0.034, up=0.002)
    jug = np.array([s * 0.026, med_lo[1] + 0.014, MED[2] + 0.012])     # jugular foramen
    def mk(id, name, kind, desc, vel=50):
        n = Nerve(f'{id}_{S}', f'{SN} {name}', 'cranial', S, 'brain / brainstem', kind, desc, vel); NV[n.d['id']] = n; return n

    n = mk('cn1', 'olfactory nerve (CN I)', 'sensory', 'Smell. Tiny fibres pass from the roof of the nose through the cribriform plate into the olfactory bulb, then the tract runs back under the frontal lobe. It is the only sense that reaches the cortex without relaying in the thalamus.', 1)
    bulb = np.array([s * 0.005, eth_hi[1] + 0.006, eth_hi[2] - 0.022])
    n.line([np.array([s * 0.006, eth_hi[1] - 0.012, eth_hi[2] - 0.018]), bulb, np.array([s * 0.008, eth_hi[1] + 0.008, eth_lo[2] + 0.004]), np.array([s * 0.014, chi[1] + 0.004, chi[2] + 0.004])], 0.0014)

    n = mk('cn2', 'optic nerve (CN II)', 'sensory', 'Vision. About a million fibres leave each retina, half cross at the optic chiasm, and the optic tract carries them to the lateral geniculate body of the thalamus on the way to the visual cortex.', 40)
    lgb = cen(S, 'lateral geniculate body')
    n.line([eye + O(ant=-0.01, lat=-0.002), apex, chi + O(lat=0.004), chi + O(lat=0.012, ant=-0.012, up=0.002), lgb], 0.0018)

    n = mk('cn3', 'oculomotor nerve (CN III)', 'motor', 'Moves the eye up, down and inward, lifts the eyelid, and constricts the pupil.', 60)
    n.line([np.array([s * 0.003, mid_lo[1] + 0.004, mid_hi[2] - 0.002]), np.array([s * 0.012, MID[1] - 0.004, mid_hi[2] + 0.016]), apex + O(up=0.002), eye + O(ant=-0.014, up=0.006, lat=-0.006)], 0.0012)
    n = mk('cn4', 'trochlear nerve (CN IV)', 'motor', 'Works one muscle, the superior oblique, which turns the eye down and in. The thinnest cranial nerve, and the only one to leave from the back of the brainstem.', 60)
    n.line([np.array([s * 0.004, mid_lo[1] + 0.003, mid_lo[2] + 0.002]), np.array([s * 0.016, mid_lo[1] + 0.002, MID[2]]), np.array([s * 0.016, MID[1], mid_hi[2] + 0.014]), apex + O(up=0.006, lat=-0.002), eye + O(ant=-0.01, up=0.012, lat=-0.008)], 0.0008)

    n = mk('cn5', 'trigeminal nerve (CN V)', 'mixed', 'The great sensory nerve of the face, with three divisions: ophthalmic (forehead and eye), maxillary (cheek and upper teeth) and mandibular (jaw and lower teeth), which also works the chewing muscles.', 55)
    gang = np.array([s * 0.024, PONS[1] + 0.004, pons_hi[2] + 0.018])
    m = n.line([np.array([s * 0.018, PONS[1], pons_hi[2] - 0.006]), gang], 0.0022)
    fr = cen('', 'frontal bone'); mx = cen(S, 'maxilla')
    n.line([gang, apex + O(up=0.008, lat=0.004), eye + O(up=0.016, ant=0.0), np.array([s * 0.028, eye[1] + 0.022, eye[2] + 0.014]), np.array([s * 0.03, eye[1] + 0.055, eye[2] + 0.004])], 0.0011, parent=m)
    n.line([gang, np.array([s * 0.026, eye[1] - 0.018, eye[2] - 0.03]), np.array([s * 0.03, eye[1] - 0.02, eye[2] - 0.004]), np.array([s * 0.028, eye[1] - 0.026, eye[2] + 0.02]), np.array([s * 0.024, eye[1] - 0.04, eye[2] + 0.022])], 0.0012, parent=m)
    side_m = manv[(manv[:, 0] * s > 0.02)]
    ram = side_m[side_m[:, 1] > np.percentile(side_m[:, 1], 70)].mean(0)
    body = side_m[(side_m[:, 2] > np.percentile(side_m[:, 2], 45)) & (side_m[:, 2] < np.percentile(side_m[:, 2], 70))].mean(0)
    chin = manv[(manv[:, 0] * s > 0.008) & (manv[:, 0] * s < 0.02)]; chin = chin[chin[:, 2].argmax()]
    v3 = n.line([gang, gang + O(up=-0.022, ant=-0.002, lat=0.006), ram + O(lat=-0.008, up=-0.006), body + O(lat=-0.004), chin + O(ant=0.002, up=0.004)], 0.0013, parent=m)
    n.line([gang + O(up=-0.022, ant=-0.002, lat=0.006), np.array([s * 0.014, tong[1] + 0.002, tong[2] - 0.012]), np.array([s * 0.008, tong[1] + 0.004, tong[2] + 0.02])], 0.0009, parent=v3, at_end=False)

    n = mk('cn6', 'abducens nerve (CN VI)', 'motor', 'Works one muscle, the lateral rectus, which turns the eye outward.', 60)
    n.line([np.array([s * 0.004, pons_lo[1] + 0.001, pons_hi[2] - 0.003]), np.array([s * 0.01, PONS[1] + 0.008, pons_hi[2] + 0.012]), apex + O(lat=0.004), eye + O(ant=-0.008, lat=0.012)], 0.0009)

    n = mk('cn7', 'facial nerve (CN VII)', 'mixed', 'Moves the muscles of facial expression, carries taste from the front two thirds of the tongue, and drives the tear and most salivary glands. Paralysis of one side is Bell palsy.', 55)
    pmj = np.array([s * 0.016, pons_lo[1] + 0.002, PONS[2] + 0.004])
    inner = np.array([s * 0.036, pons_lo[1] + 0.004, PONS[2] + 0.004])
    smf = np.array([s * 0.05, med_lo[1] + 0.018, MED[2] + 0.016])     # stylomastoid foramen
    par = np.array([s * 0.062, smf[1] - 0.004, smf[2] + 0.022])        # in the parotid gland
    m = n.line([pmj, inner, np.array([s * 0.046, inner[1] - 0.004, inner[2] - 0.002]), smf, par], 0.0014)
    zy = cen(S, 'zygomatic bone')
    for e in ([np.array([s * 0.066, eye[1] + 0.012, eye[2] - 0.012]), np.array([s * 0.05, eye[1] + 0.04, eye[2] + 0.004])],
              [np.array([s * 0.064, zy[1] + 0.004, zy[2] + 0.004]), np.array([s * 0.046, eye[1] - 0.012, eye[2] + 0.014])],
              [np.array([s * 0.058, zy[1] - 0.028, zy[2] + 0.012]), np.array([s * 0.03, mx[1] - 0.03, mx[2] + 0.03])],
              [body + O(lat=0.008, up=0.004), chin + O(lat=0.01, up=0.012, ant=0.004)],
              [np.array([s * 0.05, body[1] - 0.03, body[2] - 0.01]), np.array([s * 0.03, thy[1] + 0.004, thy[2] + 0.012])]):
        n.line([par] + e, 0.0008, parent=m)
    n = mk('cn8', 'vestibulocochlear nerve (CN VIII)', 'sensory', 'Hearing from the cochlea and balance from the vestibular organs of the inner ear.', 30)
    n.line([np.array([s * 0.044, pons_lo[1] + 0.006, PONS[2] + 0.004]), inner + O(up=0.002, ant=-0.002), pmj + O(lat=0.002, ant=-0.004)], 0.0015)
    n.inner = np.array([s * 0.044, pons_lo[1] + 0.006, PONS[2] + 0.004])

    n = mk('cn9', 'glossopharyngeal nerve (CN IX)', 'mixed', 'Taste and sensation from the back third of the tongue and the throat, the afferent limb of the gag reflex, and blood pressure sensing from the carotid sinus.', 50)
    n.line([np.array([s * 0.008, MED[1] + 0.012, MED[2] + 0.006]), jug, np.array([s * 0.026, tong[1] + 0.002, MED[2] + 0.03]), np.array([s * 0.012, tong[1] - 0.004, tong[2] - 0.022])], 0.001)

    n = mk('cn10', 'vagus nerve (CN X)', 'mixed', 'The "wanderer". Leaves the skull, runs down the neck beside the carotid artery and reaches the heart, lungs and gut as far as the colon. It carries most of the parasympathetic "rest and digest" outflow: it slows the heart and promotes digestion. It also works the voice box.', 15)
    t1y = CAN[THOR[0]]['ymid']; t5y = CAN[THOR[4]]['ymid']; t10y = CAN[THOR[9]]['ymid']
    neck_z = lambda y: zc(y) + 0.036
    m = n.line([np.array([s * 0.007, MED[1] + 0.004, MED[2] + 0.004]), jug + O(ant=-0.002), np.array([s * 0.026, CAN[CERV[2]]['ymid'], neck_z(CAN[CERV[2]]['ymid'])]),
                np.array([s * 0.024, CAN[CERV[5]]['ymid'], neck_z(CAN[CERV[5]]['ymid'])]), np.array([s * 0.022, t1y, neck_z(t1y) + 0.004]),
                np.array([s * 0.02 + 0.006, t5y, zc(t5y) + 0.05]), np.array([s * 0.008 + 0.012, t10y, zc(t10y) + 0.055]),
                np.array([(sto_lo[0] + sto_hi[0]) / 2 + s * 0.02, sto_hi[1] - 0.03, (sto_lo[2] + sto_hi[2]) / 2 + s * -0.02])], 0.0018)
    n.line([np.array([s * 0.02 + 0.006, t5y + 0.03, zc(t5y) + 0.05]), np.array([HEART[0] + s * 0.02, heart_hi[1] + 0.004, HEART[2] - 0.01]), np.array([HEART[0], heart_hi[1] - 0.012, HEART[2]])], 0.001, parent=m, at_end=False)
    loop_y = t1y - (0.05 if S == 'L' else 0.0)
    n.line([np.array([s * 0.021, loop_y, neck_z(loop_y) + 0.004]), np.array([s * 0.012, loop_y - 0.012, neck_z(loop_y) + 0.0]), np.array([s * 0.012, t1y + 0.03, thy[2] - 0.004]), np.array([s * 0.012, thy[1] - 0.012, thy[2]])], 0.0009, parent=m, at_end=False)
    n.heart = 1

    n = mk('cn11', 'accessory nerve (CN XI)', 'motor', 'Works sternocleidomastoid, which turns the head, and trapezius, which shrugs the shoulder. Its fibres arise from the upper cervical cord and loop up through the skull before leaving it.', 60)
    scv = Vx(S, 'scapula'); stop = scv[scv[:, 1].argmax()]
    n.line([canal_pt(CAN[CERV[2]]['ymid'], s * 0.006), np.array([s * 0.01, med_lo[1] + 0.004, MED[2] - 0.002]), jug + O(ant=-0.004, lat=0.002), np.array([s * 0.045, CAN[CERV[2]]['ymid'], zc(CAN[CERV[2]]['ymid']) + 0.012]),
            np.array([s * 0.06, CAN[CERV[5]]['ymid'], zc(CAN[CERV[5]]['ymid']) - 0.012]), stop + np.array([-s * 0.03, 0.012, -0.012]), stop + np.array([-s * 0.05, -0.05, -0.03])], 0.0011)

    n = mk('cn12', 'hypoglossal nerve (CN XII)', 'motor', 'Moves the tongue. If one side fails, the tongue points toward the weak side when stuck out.', 60)
    n.line([np.array([s * 0.004, MED[1] - 0.004, med_hi[2] - 0.004]), np.array([s * 0.018, med_lo[1] + 0.008, MED[2] + 0.016]), np.array([s * 0.028, tong[1] - 0.012, MED[2] + 0.036]),
            np.array([s * 0.02, tong[1] - 0.018, tong[2] - 0.016]), np.array([s * 0.008, tong[1] - 0.004, tong[2] + 0.012])], 0.0011)

for S, SN in SIDES: cranial(S, SN)

# ---------------------------------------------------------------- signal pathways
def seg(label, pts, vel, note, hl=(), syn=0.0, kind='cns', dense=False):
    p = np.array(pts, float) if dense else smooth(pts, 0.004)
    return dict(label=label, p=p, vel=vel, note=note, hl=list(hl), syn=syn, kind=kind)

def cord_run(y0, y1, x, dz=0.0):
    n = max(2, int(abs(y1 - y0) / 0.015) + 1)
    return [canal_pt(y, x) + np.array([0, 0, dz]) for y in np.linspace(y0, y1, n)]

def gyrus_pt(S, base, lat_frac, up_frac=0.75):
    """point on a gyrus: lat_frac 0 = midline, 1 = far lateral"""
    v = Vx(S, base); ax = np.abs(v[:, 0]); x = ax.min() + lat_frac * (ax.max() - ax.min())
    sel = v[np.abs(ax - x) < 0.006]
    y = np.percentile(sel[:, 1], up_frac * 100); sel = sel[np.abs(sel[:, 1] - y) < 0.006]
    return sel.mean(0)

def other(S): return 'L' if S == 'R' else 'R'
def W(S): return 'right' if S == 'R' else 'left'

def motor_down(Sbody, lat_frac, seg_name):
    """corticospinal route from the opposite motor cortex to the anterior horn of seg_name on side Sbody"""
    B = other(Sbody); sb = sgn(B); st = sgn(Sbody)
    cx = gyrus_pt(B, 'precentral gyrus', lat_frac)
    ic = cen(B, 'internal capsule') + np.array([0, 0, -0.012])
    ped = np.array([sb * 0.012, MID[1], mid_hi[2] - 0.004])
    pp = np.array([sb * 0.008, PONS[1], pons_hi[2] - 0.006])
    pyr = np.array([sb * 0.004, MED[1], med_hi[2] - 0.004])
    dec = np.array([0.0, med_lo[1] + 0.006, MED[2] + 0.004])
    ys = SEG[seg_name]['cord_y']
    a = seg('Motor cortex to brainstem', [cx, (cx + ic) / 2 + np.array([sb * 0.004, 0, 0]), ic, ped, pp, pyr], 65,
            f'Upper motor neurons in the {W(B)} precentral gyrus fire. Their axons funnel through the internal capsule and down the front of the brainstem as the corticospinal tract.',
            [f'{W(B)} precentral gyrus', f'{W(B)} internal capsule', 'peduncle of midbrain', 'pons'])
    b = seg('Crossing in the medulla', [pyr, dec, canal_pt(MED_BOT[1] - 0.012, st * 0.0035)], 65,
            f'At the pyramidal decussation about 90% of the fibres cross the midline. This is why the {W(B)} half of the brain moves the {W(Sbody)} half of the body.', ['medulla oblongata'])
    c = seg('Down the spinal cord', cord_run(MED_BOT[1] - 0.012, ys, st * 0.0035, -0.002), 65,
            f'The lateral corticospinal tract descends in the {W(Sbody)} side of the cord to segment {seg_name}.', [], kind='cord')
    return [a, b, c]

def sensory_up(Sbody, seg_name, lat_frac, fast=True):
    B = other(Sbody); sb = sgn(B); st = sgn(Sbody)
    ys = SEG[seg_name]['cord_y']
    th = cen(B, 'thalamus') + np.array([sb * 0.004, -0.004, -0.008])
    cx = gyrus_pt(B, 'postcentral gyrus', lat_frac)
    if fast:
        a = seg('Up the dorsal columns', cord_run(ys, MED_BOT[1] - 0.004, st * 0.002, -0.004), 60,
                'Fine touch fibres turn upward without synapsing and climb the dorsal columns on the same side of the cord.', [], kind='cord')
        b = seg('Relay and crossing in the medulla', [canal_pt(MED_BOT[1] - 0.004, st * 0.002) + np.array([0, 0, -0.004]), np.array([st * 0.004, med_lo[1] + 0.012, med_lo[2] + 0.006]), np.array([sb * 0.004, MED[1], MED[2]]),
                np.array([sb * 0.006, PONS[1], PONS[2]]), np.array([sb * 0.008, MID[1], MID[2]]), th], 55,
                'First relay in the dorsal column nuclei. The second neuron crosses the midline and rises through the brainstem as the medial lemniscus.', ['medulla oblongata', 'pons', 'midbrain'], syn=0.0007)
    else:
        a = seg('Crossing in the cord', [canal_pt(ys, st * 0.003) + np.array([0, 0, -0.003]), canal_pt(ys + 0.006, 0.0) + np.array([0, 0, 0.002]), canal_pt(ys + 0.014, sb * 0.004) + np.array([0, 0, 0.002])], 15,
                'Pain fibres synapse in the dorsal horn. The second neuron crosses to the other side within a segment or two.', [], syn=0.0007, kind='cord')
        b = seg('Up the spinothalamic tract', cord_run(ys + 0.014, MED_BOT[1], sb * 0.004, 0.002) + [np.array([sb * 0.008, MED[1], MED[2]]), np.array([sb * 0.012, PONS[1], PONS[2] - 0.004]), np.array([sb * 0.012, MID[1], MID[2] - 0.004]), th], 15,
                'The spinothalamic tract carries pain and temperature up the opposite side of the cord and through the brainstem.', ['medulla oblongata', 'pons', 'midbrain'], kind='cord')
    c = seg('Thalamus to sensory cortex', [th, (th + cx) / 2 + np.array([sb * 0.006, 0, 0]), cx], 50,
            f'The thalamus relays the signal to the {W(B)} postcentral gyrus, the primary somatosensory cortex, where it becomes a conscious sensation.', [f'{W(B)} thalamus', f'{W(B)} postcentral gyrus'], syn=0.0007)
    return [a, b, c]

def nerve_seg(label, nid, line, note, vel, reverse=False, syn=0.0, kind='nerve', cut=None):
    n = NV[nid]; p = n.route(line)
    if cut is not None: p = p[:cut]
    if reverse: p = p[::-1]
    return dict(label=label, p=p, vel=vel, note=note, hl=[], syn=syn, kind=kind, nerve=[nid])

def root_seg(label, S, sname, note, vel, reverse=False, syn=0.0):
    n = NV[f'root_{sname}_{S}']; p = n.dense[0]
    if reverse: p = p[::-1]
    return dict(label=label, p=p, vel=vel, note=note, hl=[], syn=syn, kind='nerve', nerve=[n.d['id']])

def plexus_route(nid, sname, line):
    """root contribution of segment sname followed by the route of `line`, trimmed of the generic first root"""
    n = NV[nid]
    S = n.d['side']
    f = foramen(S, sname)
    for i, L in enumerate(n.d['lines']):
        if L['par'] is None and np.linalg.norm(n.dense[i][0] - f) < 1e-6 and i != line:
            first = n.dense[i]; break
    r = n.route(line)
    # route(line) starts at the cords point (delay line); find where
    return np.vstack([first, r[1:]])

PATHS = []
def add(id, name, blurb, segs, cat):
    PATHS.append(dict(id=id, name=name, blurb=blurb, segs=segs, cat=cat))

# 1 voluntary finger movement
S = 'R'
med = NV['median_R']; li = [i for i, L in enumerate(med.d['lines']) if L['par'] is not None][1]   # index finger branch
segs = motor_down(S, 0.62, 'C8')
segs.append(root_seg('Anterior horn and ventral root', S, 'C8', 'The upper motor neuron synapses on a lower motor neuron in the anterior horn of C8. Its axon leaves the cord in the ventral root.', 60, syn=0.0007))
p = plexus_route('median_R', 'C8', li)
segs.append(dict(label='Brachial plexus and median nerve', p=p, vel=60, note='The fibre joins the brachial plexus behind the collarbone, then runs in the median nerve down the arm, through the carpal tunnel, to the muscles that bend the index finger.', hl=[], syn=0, kind='nerve', nerve=['median_R']))
add('move_finger', 'Bend the right index finger', 'A voluntary movement, from the left motor cortex to the right hand.', segs, 'Motor')

# 2 touch from fingertip
p = plexus_route('median_R', 'C7', li)[::-1]
segs = [dict(label='Fingertip to spinal cord', p=p, vel=55, note='Touch receptors in the fingertip fire. The signal runs up the median nerve and brachial plexus in fast, thickly myelinated fibres. The cell body sits in the dorsal root ganglion.', hl=[], syn=0, kind='nerve', nerve=['median_R']),
        root_seg('Dorsal root', S, 'C7', 'The fibre enters the cord through the dorsal root of C7.', 55, reverse=True)]
segs += sensory_up(S, 'C7', 0.62, fast=True)
add('touch_finger', 'Feel a touch on the right index finger', 'Fine touch, from skin to the left sensory cortex.', segs, 'Sensory')

# 3 knee jerk
fem = NV['femoral_R']; q = fem.quad
up = plexus_route('femoral_R', 'L4', q)[::-1]
down = plexus_route('femoral_R', 'L3', q)
y4 = SEG['L4']['cord_y']; y3 = SEG['L3']['cord_y']
segs = [dict(label='Stretch receptor to cord', p=up, vel=100, note='A tap on the patellar tendon stretches the quadriceps. Muscle spindles fire and the signal races up the femoral nerve in Ia fibres, the fastest in the body, at about 100 m/s.', hl=[], syn=0, kind='nerve', nerve=['femoral_R']),
        root_seg('Dorsal root of L4', 'R', 'L4', 'The fibre enters the cord. The lumbar segments sit high, behind the T11 to L1 vertebrae, so the root climbs the cauda equina.', 100, reverse=True),
        seg('One synapse in the cord', [canal_pt(y4, -0.004) + np.array([0, 0, -0.003]), canal_pt((y3 + y4) / 2, -0.003) + np.array([0, 0, 0.002]), canal_pt(y3, -0.0045)], 30,
            'The sensory fibre synapses directly on a motor neuron in the anterior horn. One synapse, no brain involved. That is what makes a reflex fast.', [], syn=0.001, kind='cord'),
        root_seg('Ventral root of L3', 'R', 'L3', 'The motor neuron sends its command out through the ventral root.', 60),
        dict(label='Femoral nerve to quadriceps', p=down, vel=60, note='The command returns down the femoral nerve. The quadriceps contracts and the leg kicks, about 20 to 30 thousandths of a second after the tap.', hl=[], syn=0.0008, kind='nerve', nerve=['femoral_R'])]
add('knee_jerk', 'Knee jerk reflex', 'A reflex arc that never reaches the brain.', segs, 'Reflex')

# 4 pain from the big toe
tibn = NV['tibial_L']
up = plexus_route('tibial_L', 'S1', tibn.mp)[::-1]
segs = [dict(label='Toe to spinal cord', p=up, vel=1.2, note='You stub your left big toe. Dull, burning pain travels in thin unmyelinated C fibres at only about 1 m/s, so it takes over a second to arrive. (Sharp first pain uses faster A delta fibres.)', hl=[], syn=0, kind='nerve', nerve=['tibial_L']),
        root_seg('Dorsal root of S1', 'L', 'S1', 'The fibre climbs the cauda equina to the S1 segment of the cord, behind the L1 vertebra.', 1.2, reverse=True)]
segs += sensory_up('L', 'S1', 0.06, fast=False)
add('pain_toe', 'Slow pain from the left big toe', 'Why a stubbed toe hurts a moment after you feel the knock.', segs, 'Sensory')

# 5 vision
def vision(S):
    eye = cen(S, 'sclera'); n = NV[f'cn2_{S}']; lgb = cen(S, 'lateral geniculate body'); s = sgn(S)
    occ = Vx(S, 'occipital lobe'); pole = occ[occ[:, 2].argmin()] + np.array([-s * 0.004, 0.004, 0.012])
    a = dict(label='Retina, optic nerve, chiasm, tract', p=n.dense[0], vel=20, note='Light hits the retina. Ganglion cell axons leave the eye as the optic nerve. At the chiasm fibres from the nasal half of each retina cross, so each side of the brain sees the opposite half of the visual field. The retina itself needs about 30 ms to turn light into impulses, which is most of the delay.', hl=['optic chiasm', f'{W(S)} optic nerve', f'{W(S)} optic tract'], syn=0.03, kind='nerve', nerve=[f'cn2_{S}'])
    b = seg('Optic radiation', [lgb, lgb + np.array([s * 0.016, 0.004, -0.02]), np.array([s * 0.03, lgb[1] + 0.004, (lgb[2] + pole[2]) / 2]), pole], 30,
            'After a relay in the lateral geniculate body of the thalamus, the optic radiation sweeps back to the primary visual cortex in the occipital lobe.', [f'{W(S)} lateral geniculate body', f'{W(S)} occipital lobe'], syn=0.0007)
    return [a, b]
add('vision', 'See with the right eye', 'From the retina to the visual cortex at the back of the head.', vision('R'), 'Sensory')

# 6 hearing
def hearing(S):
    s = sgn(S); n = NV[f'cn8_{S}']; B = other(S); sb = sgn(B)
    ic = cen(B, 'inferior colliculus'); mgb = cen(B, 'medial geniculate body')
    tg = Vx(B, 'posterior part of superior temporal gyrus'); tgc = tg[tg[:, 1] > np.percentile(tg[:, 1], 70)].mean(0)
    a = dict(label='Cochlea to brainstem', p=n.dense[0], vel=25, note='Hair cells in the cochlea turn sound into nerve impulses, which travel in the cochlear part of the vestibulocochlear nerve to the cochlear nuclei.', hl=[], syn=0.001, kind='nerve', nerve=[f'cn8_{S}'])
    b = seg('Crossing and climbing the brainstem', [n.dense[0][-1], np.array([0, pons_lo[1] + 0.006, PONS[2]]), np.array([sb * 0.012, PONS[1] + 0.006, PONS[2] - 0.008]), ic], 30,
            'Most fibres cross in the pons and climb as the lateral lemniscus to the inferior colliculus. Both ears reach both sides, which lets the brainstem locate sound.', ['pons', f'{W(B)} inferior colliculus'], syn=0.0014)
    c = seg('Thalamus to auditory cortex', [ic, mgb, mgb + np.array([sb * 0.016, 0.004, 0.0]), tgc], 30,
            'A relay in the medial geniculate body, then on to the primary auditory cortex on the upper surface of the temporal lobe.', [f'{W(B)} medial geniculate body', f'posterior part of {W(B)} superior temporal gyrus'], syn=0.0014)
    return [a, b, c]
add('hearing', 'Hear with the right ear', 'From the cochlea to the auditory cortex.', hearing('R'), 'Sensory')

# 7 breathing
ph = NV['phrenic_R']; main = len(ph.d['lines']) - 1
pr = np.vstack([[d for i, d in enumerate(ph.dense) if np.linalg.norm(d[0] - foramen('R', 'C4')) < 1e-6][0], ph.dense[main][1:]])
segs = [seg('Breathing centre to cervical cord', [np.array([-0.004, MED[1] - 0.004, MED[2]])] + cord_run(MED_BOT[1], SEG['C4']['cord_y'], -0.003, 0.002), 60,
            'Rhythm generating neurons in the medulla fire about 12 to 16 times a minute without any conscious effort and send the command down to segments C3 to C5.', ['medulla oblongata'], kind='cord'),
        root_seg('Ventral root of C4', 'R', 'C4', 'Phrenic motor neurons in the anterior horn send their axons out of the cord.', 60, syn=0.0007),
        dict(label='Phrenic nerve to diaphragm', p=pr, vel=60, note='The phrenic nerve runs down the neck and through the chest beside the heart. The diaphragm contracts and flattens, and air is drawn into the lungs.', hl=['diaphragm'], syn=0, kind='nerve', nerve=['phrenic_R'])]
add('breathe', 'Take a breath', 'The automatic rhythm that runs the diaphragm.', segs, 'Autonomic')

# 8 vagus slows the heart
vg = NV['cn10_R']
segs = [dict(label='Medulla to heart', p=vg.route(1), vel=12, note='Parasympathetic neurons in the medulla send signals down the vagus nerve. Near the heart they relay in small ganglia and release acetylcholine on the pacemaker, slowing the heart rate.', hl=['medulla oblongata', 'wall of ventricle', 'wall of right atrium'], syn=0.002, kind='nerve', nerve=['cn10_R'])]
add('vagus_heart', 'Slow the heart (vagus nerve)', 'Rest and digest: the brake on the heart.', segs, 'Autonomic')

# 9 lift the foot
cf = NV['common_fibular_L']
segs = motor_down('L', 0.04, 'L5')
segs.append(root_seg('Ventral root of L5', 'L', 'L5', 'The lower motor neuron leaves the cord and descends in the cauda equina to its exit below the L5 vertebra.', 55, syn=0.0007))
segs.append(dict(label='Sciatic and deep fibular nerves', p=plexus_route('common_fibular_L', 'L5', cf.dp), vel=50, note='The fibre runs in the sciatic nerve down the back of the thigh, around the neck of the fibula in the common fibular nerve, then down the front of the leg in the deep fibular nerve to the muscles that lift the foot and toes. At about a metre, these are among the longest cells in the body.', hl=[], syn=0, kind='nerve', nerve=['sciatic_L', 'common_fibular_L']))
add('lift_foot', 'Lift the left foot', 'The longest motor route in the body, about 1.5 metres from cortex to shin.', segs, 'Motor')

# 10 smile
f7 = NV['cn7_L']
fl = [i for i, L in enumerate(f7.d['lines']) if L['par'] is not None][2]
cxp = gyrus_pt('R', 'precentral gyrus', 0.9)
ic = cen('R', 'internal capsule')
segs = [seg('Motor cortex to facial nucleus', [cxp, ic, np.array([-0.012, MID[1], mid_hi[2] - 0.004]), np.array([-0.004, PONS[1] + 0.006, PONS[2] + 0.004]), np.array([0.008, pons_lo[1] + 0.004, PONS[2] - 0.004]), f7.dense[0][0]], 60,
            'The face area lies low on the side of the motor cortex. Fibres descend through the internal capsule and cross in the pons to the facial nucleus.', ['right precentral gyrus', 'right internal capsule', 'pons']),
        dict(label='Facial nerve to the cheek', p=f7.route(fl), vel=55, note='The facial nerve passes through the temporal bone, leaves the skull below the ear, fans out inside the parotid gland and reaches the muscles that lift the corner of the mouth.', hl=[], syn=0.0007, kind='nerve', nerve=['cn7_L'])]
add('smile', 'Smile (left side)', 'From the face area of the motor cortex along the facial nerve.', segs, 'Motor')

# ---------------------------------------------------------------- meshes
MUSCLE_IN_SKEL = re.compile(r'fibularis|levator scapulae|subscapularis|iliotibial|tibialis')
def cls(p):
    side, base = split(p['name'])
    sy = p['system']
    if sy == 'skeletal':
        if MUSCLE_IN_SKEL.search(base): return None
        if 'tooth' in base: return 'tooth'
        if 'cartilage' in base: return 'cartilage'
        if 'disk' in base: return 'disc'
        if 'gingiva' in base: return None
        return 'bone'
    if 'lacrimal bone' in base: return 'bone'
    if sy == 'nervous':
        if 'tentorium' in base: return None
        if re.search(r'nerve|ganglion', base): return 'nervemesh'
        return 'brain'
    if base == 'skin': return 'skin'
    if base in ('wall of ventricle', 'wall of left atrium', 'wall of right atrium', 'diaphragm', 'stomach', 'sclera', 'cornea', 'iris', 'tongue', 'external ear'): return 'organ'
    return None

def region(base):
    b = base
    if re.search(r'frontal|parietal|occipital|temporal|sphenoid|ethmoid', b): return 'Skull, braincase'
    if re.search(r'mandible|maxilla|zygomatic|nasal|palatine|vomer|lacrimal', b): return 'Skull, facial skeleton'
    if 'hyoid' in b: return 'Neck'
    if re.search(r'cervical|atlas|axis', b): return 'Spine, cervical'
    if 'thoracic' in b: return 'Spine, thoracic'
    if 'lumbar' in b: return 'Spine, lumbar'
    if 'sacrum' in b: return 'Spine, sacrum and coccyx'
    if re.search(r'\brib\b|sternum|manubrium|xiphoid|costal', b): return 'Thoracic cage'
    if re.search(r'clavicle|scapula', b): return 'Shoulder girdle'
    if re.search(r'humerus|radius|ulna', b): return 'Arm and forearm'
    if re.search(r'scaphoid|lunate|triquetral|pisiform|trapezium|trapezoid|capitate|hamate', b): return 'Wrist (carpals)'
    if re.search(r'metacarpal|finger|thumb', b): return 'Hand'
    if 'hip bone' in b: return 'Pelvic girdle'
    if re.search(r'femur|patella|tibia|fibula', b): return 'Thigh and leg'
    if re.search(r'talus|calcaneus|navicular|cuboid|cuneiform', b): return 'Ankle (tarsals)'
    if re.search(r'metatarsal|toe|sesamoid', b): return 'Foot'
    if re.search(r'arytenoid|corniculate|cricoid|thyroid|cuneiform cartilage', b): return 'Larynx'
    if 'alar' in b: return 'Nose'
    if 'disk' in b: return 'Spine'
    return ''

parts = []; blob = bytearray()
counts = {}
for p in A['parts']:
    k = cls(p)
    if not k: continue
    v, ix = mesh(p)
    lo, hi = v.min(0), v.max(0); ext = np.maximum(hi - lo, 1e-9)
    q = np.round((v - lo) / ext * 65535).astype('<u2')
    while len(blob) % 4: blob.append(0)
    po = len(blob); blob += q.tobytes()
    big = len(v) > 65535
    while len(blob) % 4: blob.append(0)
    io = len(blob); blob += ix.astype('<u4' if big else '<u2').tobytes()
    side, base = split(p['name'])
    parts.append(dict(n=p['name'], k=k, s=side, fma=p['conceptId'], g=region(base) if k in ('bone', 'tooth', 'cartilage', 'disc') else '',
                      po=po, vc=len(v), io=io, ic=len(ix), big=int(big), lo=[round(float(x), 5) for x in lo], hi=[round(float(x), 5) for x in hi]))
    counts[k] = counts.get(k, 0) + 1
open(f'{OUT}/anatomy.bin', 'wb').write(bytes(blob))

def rp(p): return [round(float(x), 4) for x in np.asarray(p).reshape(-1)]
nerves = []
for n in NERVES:
    d = dict(n.d)
    d['lines'] = [dict(p=rp(L['p']), r=L['r'], d=round(L['d'], 4)) for L in n.d['lines']]
    if 'ganglia' in d: d['ganglia'] = rp(np.array(d['ganglia']))
    d['len'] = round(max(L['d'] + plen(L['p']) for L in n.d['lines']), 3)
    nerves.append(d)
paths = []
for P_ in PATHS:
    segs = []
    for sgm in P_['segs']:
        segs.append(dict(label=sgm['label'], p=rp(sgm['p']), vel=sgm['vel'], note=sgm['note'], hl=sgm['hl'], syn=sgm['syn'], kind=sgm['kind'], nerve=sgm.get('nerve', []), len=round(plen(sgm['p']), 4)))
    paths.append(dict(id=P_['id'], name=P_['name'], blurb=P_['blurb'], cat=P_['cat'], segs=segs))

json.dump(dict(parts=parts, nerves=nerves, paths=paths, bytes=len(blob)), open(f'{OUT}/data.json', 'w'), separators=(',', ':'))
print('parts', counts, 'bin MB', round(len(blob) / 1e6, 2), 'json MB', round(os.path.getsize(f'{OUT}/data.json') / 1e6, 2))
print('nerves', len(NERVES), 'paths', len(PATHS))
for P_ in paths:
    t = sum(s['len'] / s['vel'] + s['syn'] for s in P_['segs'])
    print(f"  {P_['name']:42s} {sum(s['len'] for s in P_['segs']):.2f} m  {t*1000:.1f} ms")
