"""
LiDAR Visualize Tool  v2.0
==========================
Three-tab GUI for inspecting merged point cloud maps.

  Tab 1 — Viewer
      Load any .pcd, height colormap, optional trajectory overlay,
      optional statistical denoising.

  Tab 2 — Editor
      Manual crop-select deletion  (K=enter select, C=confirm, close)
      Trajectory auto-band removal  (mark & delete points near path)
      Undo / save.

  Tab 3 — Analysis
      Floor / Ceiling / Wall classification (coloured)
      Door detection (gold wireframe boxes)
      Room detection (2D floor-plan connected components)
      Interactive viewer: F/C/W/O/D/R/H keys to toggle layers

Requirements:  pip install open3d numpy scipy
"""

import os, threading, tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# ── lazy imports ──────────────────────────────────────────────────────────────
o3d = None; Rotation = None; ndimage = None

def _libs():
    global o3d, Rotation, ndimage
    if o3d      is None:
        import open3d as _o;                        o3d      = _o
    if Rotation is None:
        from scipy.spatial.transform import Rotation as _R; Rotation = _R
    if ndimage  is None:
        from scipy import ndimage as _nd;           ndimage  = _nd


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def colorize_height(pcd):
    pts = np.asarray(pcd.points)
    t = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-9)
    r = np.clip(2*t - 1, 0, 1); g = np.clip(2*t, 0, 1); b = np.clip(1 - 2*t, 0, 1)
    out = o3d.geometry.PointCloud(pcd)
    out.colors = o3d.utility.Vector3dVector(np.stack([r, g, b], 1))
    return out

def apply_denoise(pcd, strength, nb):
    if strength == 0 or len(pcd.points) < nb + 1: return pcd
    cl, _ = pcd.remove_statistical_outlier(int(nb), 3.0 / strength)
    return cl

def build_trajectory(traj_file, sphere_radius=0.15, step=1):
    """Load trajectory txt (x y z per line) and build Open3D geometry."""
    data = np.loadtxt(traj_file, comments="#")
    if data.ndim == 1: data = data.reshape(1, -1)
    if data.shape[1] == 3:
        pts = data.astype(np.float64)
    else:
        raise ValueError(f"Trajectory file must have 3 columns (x y z), got {data.shape[1]}")
    N = len(pts); geoms = []
    if N > 1:
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(N-1)])
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.15, 0.15]] * (N-1))
        geoms.append(ls)
    idx = list(range(0, N, step))
    if N > 0 and idx[-1] != N-1: idx.append(N-1)
    for i in idx:
        t = i / max(N-1, 1); col = [1.0, 0.3*(1-t), 0.0]
        sp = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sp.translate(pts[i]); sp.paint_uniform_color(col); sp.compute_vertex_normals()
        geoms.append(sp)
    if N > 0:
        for pt, col in [(pts[0], [0.0, 1.0, 0.2]), (pts[-1], [1.0, 0.0, 0.0])]:
            sp = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius * 2)
            sp.translate(pt); sp.paint_uniform_color(col); sp.compute_vertex_normals()
            geoms.append(sp)
    return geoms, pts

def remove_traj_band(pcd, traj_pts, radius):
    if not len(traj_pts) or radius <= 0: return pcd
    tp = o3d.geometry.PointCloud()
    tp.points = o3d.utility.Vector3dVector(traj_pts.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(tp); pts = np.asarray(pcd.points)
    keep = np.ones(len(pts), dtype=bool); r2 = radius * radius
    for i, pt in enumerate(pts):
        k, _, d = tree.search_radius_vector_3d(pt, radius)
        if k > 0 and d[0] <= r2: keep[i] = False
    return pcd.select_by_index(np.where(keep)[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

COL_FLOOR   = [0.60, 0.38, 0.14]; COL_CEILING = [0.88, 0.88, 0.88]
COL_OTHER   = [0.45, 0.60, 0.75]; COL_DOOR    = [1.00, 0.85, 0.00]
WALL_PAL    = [[0.95,0.38,0.38],[0.38,0.88,0.48],[0.38,0.58,0.95],
               [0.95,0.65,0.18],[0.80,0.38,0.90],[0.18,0.85,0.85],
               [0.90,0.90,0.28],[0.50,0.90,0.78]]
ROOM_COLORS = [[0.12,0.47,0.71],[0.20,0.63,0.17],[0.89,0.47,0.76],
               [0.58,0.40,0.74],[0.55,0.34,0.29],[0.09,0.75,0.81],
               [0.74,0.74,0.13],[0.00,0.50,0.50],[0.75,0.25,0.00],
               [0.25,0.00,0.75],[0.00,0.75,0.25],[0.75,0.00,0.25]]

def _n(model):
    a,b,c,d = model; n = np.array([a,b,c], dtype=float); return n / np.linalg.norm(n)

def classify_planes(pcd, ransac_dist=0.05, max_planes=20, min_ratio=0.01,
                    ceil_top_pct=15, ceil_merge_dist=0.6,
                    wall_angle_lo=70, wall_angle_hi=110, wall_min_span=1.2, log=print):
    pts_z = np.asarray(pcd.points)[:, 2]
    z_ceil_th = np.percentile(pts_z, 100 - ceil_top_pct)
    total = len(pcd.points); remaining = o3d.geometry.PointCloud(pcd)
    raw_h = []; raw_v = []
    for k in range(max_planes):
        if len(remaining.points) < 50: break
        model, inliers = remaining.segment_plane(ransac_dist, 3, 2000)
        if len(inliers) / total < min_ratio: break
        pln = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        pts = np.asarray(pln.points); z_mid = pts[:, 2].mean(); z_span = pts[:, 2].ptp()
        n = _n(model); z_al = abs(n[2])
        log(f"  Plane {k+1}: z={z_mid:.2f} span={z_span:.2f} z_al={z_al:.3f} {len(inliers):,}pts")
        if z_al > 0.80: raw_h.append((pln, model, z_mid))
        else:           raw_v.append((pln, model, z_mid, z_span))

    upper_h = [(p,m,z) for p,m,z in raw_h if z >= z_ceil_th]
    lower_h = [(p,m,z) for p,m,z in raw_h if z < z_ceil_th]
    ceil_pcds = []; ceil_ref_model = None
    if upper_h:
        upper_h.sort(key=lambda x: -len(x[0].points))
        ref_p, ref_m, ref_z = upper_h[0]; ceil_ref_model = ref_m; ceil_pcds.append(ref_p)
        for p, m, z in upper_h[1:]:
            dist  = abs(ref_z - z)
            angle = np.degrees(np.arccos(np.clip(abs(np.dot(_n(ref_m), _n(m))), -1, 1)))
            if dist <= ceil_merge_dist and angle < 20: ceil_pcds.append(p)
            else: remaining += p
    lower_h.sort(key=lambda x: x[2])
    floor_pcd = None; floor_z = np.percentile(pts_z, 5); ceil_z = np.percentile(pts_z, 95)
    if lower_h: floor_pcd = lower_h[0][0]; floor_z = lower_h[0][2]
    for p, m, z in lower_h[1:]: remaining += p
    ceil_z = upper_h[0][2] if upper_h else ceil_z
    ceil_n = _n(ceil_ref_model) if ceil_ref_model is not None else np.array([0., 0., 1.])
    wall_pcds = []; wall_models = []
    for p, m, z_mid, z_span in raw_v:
        n = _n(m); angle = np.degrees(np.arccos(np.clip(abs(np.dot(n, ceil_n)), -1, 1)))
        dev = 90 - angle
        if abs(dev) <= (90 - wall_angle_lo) and z_span >= wall_min_span:
            wall_pcds.append(p); wall_models.append(m)
            log(f"  → Wall: z_span={z_span:.2f} angle_dev={dev:.1f}°")
        else: remaining += p
    log(f"\n  floor={'yes' if floor_pcd else 'no'}  ceil_planes={len(ceil_pcds)}  walls={len(wall_pcds)}")
    return {'floor': floor_pcd, 'ceiling_pcds': ceil_pcds, 'walls': wall_pcds,
            'wall_models': wall_models, 'other': remaining,
            'floor_z': floor_z, 'ceil_z': ceil_z, 'ceil_normal': ceil_n}

def build_struct_layers(struct):
    layers = {}
    if struct['floor']:
        f = o3d.geometry.PointCloud(struct['floor']); f.paint_uniform_color(COL_FLOOR); layers['floor'] = f
    if struct['ceiling_pcds']:
        m = o3d.geometry.PointCloud()
        for cp in struct['ceiling_pcds']: m += cp
        m.paint_uniform_color(COL_CEILING); layers['ceiling'] = m
    walls = []
    for i, wp in enumerate(struct['walls']):
        c = o3d.geometry.PointCloud(wp); c.paint_uniform_color(WALL_PAL[i % len(WALL_PAL)]); walls.append(c)
    layers['walls'] = walls
    if struct['other'] and len(struct['other'].points) > 0:
        o_ = o3d.geometry.PointCloud(struct['other']); o_.paint_uniform_color(COL_OTHER); layers['other'] = o_
    return layers

def detect_doors(wall_pcds, wall_models, floor_z, ceil_z,
                 grid_res=0.05, door_min_w=0.60, door_max_w=2.20,
                 door_min_h=1.80, door_max_h=3.20, log=print):
    doors = []
    for wi, (wp, model) in enumerate(zip(wall_pcds, wall_models)):
        pts = np.asarray(wp.points)
        if len(pts) < 20: continue
        a,b,c,d = model; n = np.array([a,b,c], dtype=float); n /= np.linalg.norm(n)
        up = np.array([0., 0., 1.]); u = np.cross(n, up); u /= np.linalg.norm(u)
        pu = pts @ u; pv = pts[:, 2]; pw = pts @ n
        u_min, u_max = pu.min()-0.1, pu.max()+0.1; v_min, v_max = pv.min()-0.1, pv.max()+0.1
        W = max(2, int((u_max-u_min)/grid_res)); H = max(2, int((v_max-v_min)/grid_res))
        iu = np.clip(((pu-u_min)/grid_res).astype(int), 0, W-1)
        iv = np.clip(((pv-v_min)/grid_res).astype(int), 0, H-1)
        occ = np.zeros((H, W), dtype=np.uint8); occ[iv, iu] = 1
        st = ndimage.generate_binary_structure(2, 1)
        occ = ndimage.binary_closing(occ, structure=st, iterations=3)
        labeled, n_lbl = ndimage.label(~occ)
        for lbl in range(1, n_lbl+1):
            rows, cols = np.where(labeled == lbl)
            if not len(rows): continue
            r0, r1 = rows.min(), rows.max(); c0, c1 = cols.min(), cols.max()
            h_m = (r1-r0+1)*grid_res; w_m = (c1-c0+1)*grid_res
            if pv.min() + r0*grid_res > floor_z + 0.4: continue
            if door_min_h <= h_m <= door_max_h and door_min_w <= w_m <= door_max_w:
                depth = pw.mean(); dh = max(0.15, pw.std()*2+0.05)
                u_c = u_min+(c0+c1+1)/2*grid_res; z_c = v_min+(r0+r1+1)/2*grid_res
                hu = w_m/2; hz = h_m/2; ctr = u_c*u+depth*n; ctr[2] += z_c
                C = []
                for su in [-hu, hu]:
                    for sz in [-hz, hz]:
                        for sd in [-dh, dh]:
                            pt = ctr+su*u+sd*n; pt = pt.copy(); pt[2] += sz; C.append(pt)
                edges = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]]
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(np.array(C, dtype=float))
                ls.lines  = o3d.utility.Vector2iVector(edges)
                ls.colors = o3d.utility.Vector3dVector([COL_DOOR] * len(edges))
                doors.append(ls); log(f"  Wall {wi+1}: door {w_m:.2f}×{h_m:.2f}m")
    log(f"  → {len(doors)} door(s)"); return doors

def detect_rooms(pcd, traj_pts=None, traj_band=0.5,
                 slice_z_lo_pct=20, slice_z_hi_pct=70,
                 grid_res=0.10, close_iter=4, min_room_m2=1.0, log=print):
    working = o3d.geometry.PointCloud(pcd)
    if traj_pts is not None and len(traj_pts) > 0 and traj_band > 0:
        before = len(working.points)
        working = remove_traj_band(working, traj_pts, traj_band)
        log(f"  Traj strip: {before:,}→{len(working.points):,}")
    pts = np.asarray(working.points)
    if not len(pts): return colorize_height(pcd), 0, []
    z = pts[:, 2]; z_lo = np.percentile(z, slice_z_lo_pct); z_hi = np.percentile(z, slice_z_hi_pct)
    mask = (z >= z_lo) & (z <= z_hi)
    if mask.sum() < 100: mask = np.ones(len(pts), dtype=bool)
    xy = pts[mask, :2]
    x_min, y_min = xy.min(axis=0) - 1.0; x_max, y_max = xy.max(axis=0) + 1.0
    W = int((x_max-x_min)/grid_res)+1; H = int((y_max-y_min)/grid_res)+1
    log(f"  Grid:{W}×{H} res={grid_res}m {mask.sum():,}pts")
    ix = np.clip(((xy[:,0]-x_min)/grid_res).astype(int), 0, W-1)
    iy = np.clip(((xy[:,1]-y_min)/grid_res).astype(int), 0, H-1)
    occ = np.zeros((H, W), dtype=bool); occ[iy, ix] = True
    st = ndimage.generate_binary_structure(2, 2)
    occ = ndimage.binary_closing(occ, structure=st, iterations=close_iter)
    occ = ndimage.binary_dilation(occ, structure=st, iterations=2)
    labeled, n_raw = ndimage.label(~occ)
    min_cells = max(1, int(min_room_m2 / grid_res**2))
    room_map = {}; room_areas = {}; rid = 0
    for lbl in range(1, n_raw+1):
        cnt = (labeled == lbl).sum()
        if cnt >= min_cells: rid += 1; room_map[lbl] = rid; room_areas[rid] = cnt * grid_res**2
    n_rooms = rid; log(f"  Rooms: {n_rooms}")
    for i, a in room_areas.items(): log(f"    Room {i}: {a:.1f}m²")
    if not n_rooms: return colorize_height(pcd), 0, []
    all_pts = np.asarray(pcd.points); colors = np.full((len(all_pts), 3), 0.25)
    ix_all = np.clip(((all_pts[:,0]-x_min)/grid_res).astype(int), 0, W-1)
    iy_all = np.clip(((all_pts[:,1]-y_min)/grid_res).astype(int), 0, H-1)
    cl = labeled[iy_all, ix_all]
    for gl, ri in room_map.items(): colors[cl == gl] = ROOM_COLORS[(ri-1) % len(ROOM_COLORS)]
    out = o3d.geometry.PointCloud(pcd); out.colors = o3d.utility.Vector3dVector(colors)
    return out, n_rooms, list(room_areas.values())


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

BG="#f0f4f8"; CARD="#ffffff"; ACCENT="#2563eb"; DANGER="#dc2626"; SUCCESS="#16a34a"
TEXT="#1e293b"; SUBTEXT="#64748b"; BORDER="#cbd5e1"; ACC_HV="#1d4ed8"
FH1=("Segoe UI",15,"bold"); FH2=("Segoe UI",11,"bold")
FB=("Segoe UI",10); FM=("Consolas",9); FS=("Segoe UI",9)

def sbtn(p, text, cmd, bg=ACCENT, fg="white", w=None):
    kw = dict(text=text, command=cmd, bg=bg, fg=fg,
              font=("Segoe UI",10,"bold"), relief="flat",
              activebackground=ACC_HV, activeforeground="white",
              cursor="hand2", padx=14, pady=6, bd=0)
    if w: kw["width"] = w
    return tk.Button(p, **kw)

def card(p, **kw):
    return tk.Frame(p, bg=CARD, relief="flat",
                    highlightbackground=BORDER, highlightthickness=1, **kw)

def sep(p, row, cols=3):
    ttk.Separator(p, orient="horizontal").grid(
        row=row, column=0, columnspan=cols, sticky="ew", pady=5)

def slider_row(parent, row, label, var, lo, hi, res, val_lbl, fmt, length=300, col=0):
    tk.Label(parent, text=label, font=FB, bg=CARD, fg=TEXT).grid(
        row=row, column=col, sticky="w", pady=2)
    def _cmd(v): val_lbl.config(text=fmt(v))
    tk.Scale(parent, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
             variable=var, length=length, command=_cmd, showvalue=False,
             bg=CARD, troughcolor=BORDER, highlightthickness=0).grid(
        row=row, column=col+1, padx=6)
    val_lbl.grid(row=row, column=col+2, sticky="w")

def scrollable(parent):
    canvas = tk.Canvas(parent, bg=BG, highlightthickness=0)
    sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side=tk.RIGHT, fill=tk.Y); canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    f = tk.Frame(canvas, bg=BG); wid = canvas.create_window((0,0), window=f, anchor="nw")
    f.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: (
        canvas.configure(scrollregion=canvas.bbox("all")),
        canvas.itemconfig(wid, width=e.width)))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    return canvas, f

def logbox(parent):
    txt = tk.Text(parent, height=8, font=FM, bg="#0f172a", fg="#94a3b8",
                  relief="flat", state=tk.DISABLED, wrap=tk.WORD)
    sc = ttk.Scrollbar(parent, command=txt.yview); txt.config(yscrollcommand=sc.set)
    sc.pack(side=tk.RIGHT, fill=tk.Y); txt.pack(fill=tk.BOTH, expand=True)
    def _log(msg):
        def _do():
            txt.config(state=tk.NORMAL)
            txt.insert(tk.END, msg + "\n"); txt.see(tk.END)
            txt.config(state=tk.DISABLED)
        txt.winfo_toplevel().after(0, _do)
    return txt, _log


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

class ViewerTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG); self._pcd = None; self._path = None; self._build()

    def _build(self):
        _, f = scrollable(self)
        tk.Label(f, text="Point Cloud Viewer", font=FH1, bg=BG, fg=TEXT).pack(
            anchor="w", padx=24, pady=(18,6))

        # file
        fc = card(f, padx=16, pady=12); fc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(fc, text="PCD File", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self.file_var = tk.StringVar()
        tk.Label(fc, text="File:", font=FB, bg=CARD, fg=TEXT).grid(row=1, column=0, sticky="w")
        tk.Entry(fc, textvariable=self.file_var, width=52, font=FB, relief="solid", bd=1).grid(
            row=1, column=1, padx=6)
        sbtn(fc, "Browse", self._browse_pcd, bg="#475569").grid(row=1, column=2)
        fc.columnconfigure(1, weight=1)
        self.info_lbl = tk.Label(fc, text="", font=FS, bg=CARD, fg=SUBTEXT)
        self.info_lbl.grid(row=2, columnspan=3, sticky="w", pady=(4,0))

        # denoise
        dc = card(f, padx=16, pady=10); dc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(dc, text="Denoise", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self.dn_s = tk.DoubleVar(value=0); self.dn_n = tk.IntVar(value=20)
        l_ds = tk.Label(dc, text="OFF",  width=9, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)
        l_dn = tk.Label(dc, text="20",   width=9, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)
        slider_row(dc, 1, "Strength (0=off):", self.dn_s, 0, 10, 0.5, l_ds,
                   lambda v: "OFF" if float(v)==0 else f"{float(v):.1f}")
        slider_row(dc, 2, "Neighbour count:", self.dn_n, 5, 50, 5, l_dn,
                   lambda v: str(int(float(v))))

        # trajectory
        tc = card(f, padx=16, pady=10); tc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(tc, text="Trajectory", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self.traj_on   = tk.BooleanVar(value=False); self.traj_file = tk.StringVar()
        self.traj_r    = tk.DoubleVar(value=0.15);   self.traj_step = tk.IntVar(value=5)
        tk.Checkbutton(tc, text="Show trajectory (red/orange spheres)", variable=self.traj_on,
                       font=FB, bg=CARD, fg=TEXT,
                       activebackground=CARD, selectcolor="#dbeafe").grid(
            row=1, columnspan=3, sticky="w")
        l_tr = tk.Label(tc, text="0.15 m", width=9, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)
        l_ts = tk.Label(tc, text="every 5", width=9, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)
        slider_row(tc, 2, "Sphere radius (m):", self.traj_r,   0.05, 1.0, 0.05, l_tr,
                   lambda v: f"{float(v):.2f} m")
        slider_row(tc, 3, "Sphere interval:",   self.traj_step, 1,  20,  1,    l_ts,
                   lambda v: f"every {int(float(v))}")
        tk.Label(tc, text="Trajectory file:", font=FB, bg=CARD, fg=TEXT).grid(
            row=4, column=0, sticky="w", pady=(6,0))
        tk.Entry(tc, textvariable=self.traj_file, width=44, font=FB, relief="solid", bd=1).grid(
            row=4, column=1, padx=6, pady=(6,0))
        sbtn(tc, "Browse", lambda: self._browse_file(self.traj_file), bg="#475569").grid(
            row=4, column=2, pady=(6,0))
        tc.columnconfigure(1, weight=1)

        ac = tk.Frame(f, bg=BG); ac.pack(fill=tk.X, padx=24, pady=10)
        sbtn(ac, "▶  Open Viewer", self._open, w=16).pack(side=tk.LEFT)
        sbtn(ac, "⟳  Reset", self._reset, bg="#475569", w=10).pack(side=tk.LEFT, padx=8)
        self.status = tk.Label(ac, text="", font=FS, bg=BG, fg=SUBTEXT)
        self.status.pack(side=tk.LEFT, padx=8)

    def _browse_pcd(self):
        p = filedialog.askopenfilename(filetypes=[("PCD","*.pcd"),("All","*.*")])
        if p: self.file_var.set(p); self._load(p)

    def _browse_file(self, var):
        p = filedialog.askopenfilename(filetypes=[("TXT","*.txt"),("All","*.*")])
        if p: var.set(p)

    def _load(self, path):
        _libs(); self.status.config(text="Loading …"); self.update()
        self._pcd = o3d.io.read_point_cloud(path); self._path = path
        pts = np.asarray(self._pcd.points); z = pts[:, 2]
        self.info_lbl.config(text=(
            f"{len(pts):,} pts  "
            f"X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
            f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]  "
            f"Z[{z.min():.2f},{z.max():.2f}]"))
        self.status.config(text="Loaded.")

    def _reset(self):
        self.dn_s.set(0); self.dn_n.set(20); self.traj_on.set(False)
        self.status.config(text="Reset.")

    def _open(self):
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Select a PCD file"); return
        _libs()
        if path != self._path or self._pcd is None: self._load(path)
        self.status.config(text="Processing …"); self.update()
        pcd = o3d.geometry.PointCloud(self._pcd)
        if self.dn_s.get() > 0: pcd = apply_denoise(pcd, self.dn_s.get(), self.dn_n.get())
        if not pcd.has_colors(): pcd = colorize_height(pcd)
        geoms = [pcd]
        if self.traj_on.get():
            tf = self.traj_file.get().strip()
            if tf and os.path.isfile(tf):
                try:
                    tg, _ = build_trajectory(tf, self.traj_r.get(), self.traj_step.get())
                    geoms.extend(tg)
                except Exception as e:
                    messagebox.showwarning("Trajectory", str(e))
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        self.status.config(text=f"Showing {len(np.asarray(pcd.points)):,} pts")
        threading.Thread(target=o3d.visualization.draw_geometries, args=(geoms,),
                         kwargs=dict(window_name=os.path.basename(path),
                                     width=1280, height=800), daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — EDITOR
# ═══════════════════════════════════════════════════════════════════════════════

class EditorTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._pcd = None; self._path = None
        self._undo_stack = []; self._sel_idx = np.array([], dtype=int)
        self._build()

    def _build(self):
        _, f = scrollable(self)
        tk.Label(f, text="Point Cloud Editor", font=FH1, bg=BG, fg=TEXT).pack(
            anchor="w", padx=24, pady=(18,6))

        # file
        fc = card(f, padx=16, pady=12); fc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(fc, text="File", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self.file_var = tk.StringVar(); self.save_var = tk.StringVar()
        for r, (lbl, var, kind) in enumerate([
                ("PCD file:", self.file_var, "pcd"),
                ("Save as:",  self.save_var, "save")], 1):
            tk.Label(fc, text=lbl, font=FB, bg=CARD, fg=TEXT).grid(row=r, column=0, sticky="w", pady=3)
            tk.Entry(fc, textvariable=var, width=46, font=FB, relief="solid", bd=1).grid(
                row=r, column=1, padx=6)
            sbtn(fc, "Browse", (lambda k=kind, v=var: self._browse(v, k)),
                 bg="#475569").grid(row=r, column=2)
        fc.columnconfigure(1, weight=1)
        self.info_lbl = tk.Label(fc, text="", font=FS, bg=CARD, fg=SUBTEXT)
        self.info_lbl.grid(row=3, columnspan=3, sticky="w", pady=(4,0))

        # instructions
        ic = card(f, padx=16, pady=10); ic.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(ic, text="Manual Selection", font=FH2, bg=CARD, fg=TEXT).pack(anchor="w", pady=(0,6))
        for s in [
            "1. Load PCD  →  click Load Cloud",
            "2. Click  Open Selection Viewer",
            "3. Press  K  to enter crop/select mode",
            "4. Drag to draw bounding box around points to REMOVE",
            "5. Press  C  to confirm selection",
            "6. Close the viewer window",
            "7. Click  Delete Selected  (highlighted in red in Preview)",
            "8. Repeat 2-7 as needed  |  Undo to revert  |  Save when done"]:
            tk.Label(ic, text=s, font=FS, bg=CARD, fg=TEXT, anchor="w").pack(anchor="w")

        # trajectory auto-select
        tc = card(f, padx=16, pady=10); tc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(tc, text="Trajectory Auto-Select", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        tk.Label(tc, text="Load trajectory → auto-mark points within band → review → delete",
                 font=FS, bg=CARD, fg=SUBTEXT).grid(row=1, columnspan=3, sticky="w", pady=(0,6))
        self.traj_file = tk.StringVar(); self.band_r = tk.DoubleVar(value=0.5)
        tk.Label(tc, text="Trajectory file:", font=FB, bg=CARD, fg=TEXT).grid(
            row=2, column=0, sticky="w")
        tk.Entry(tc, textvariable=self.traj_file, width=40, font=FB, relief="solid", bd=1).grid(
            row=2, column=1, padx=6)
        sbtn(tc, "Browse", lambda: self._browse(self.traj_file, "traj"),
             bg="#475569").grid(row=2, column=2)
        l_br = tk.Label(tc, text="0.50 m", width=9, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)
        slider_row(tc, 3, "Band radius (m):", self.band_r, 0.1, 3.0, 0.1, l_br,
                   lambda v: f"{float(v):.1f} m")
        sbtn(tc, "Auto-Select Trajectory Band", self._auto_select,
             bg="#7c3aed").grid(row=4, column=0, columnspan=3, sticky="w", pady=(8,0))
        tc.columnconfigure(1, weight=1)

        # actions
        ac = tk.Frame(f, bg=BG); ac.pack(fill=tk.X, padx=24, pady=8)
        btns = [
            ("⬆ Load Cloud",          self._load,    ACCENT,    14),
            ("⬡ Open Selection Viewer",self._pick,    "#7c3aed", 20),
            ("👁 Preview",             self._preview, "#0891b2", 10),
            ("✕ Delete Selected",      self._delete,  DANGER,    16),
            ("↩ Undo",                 self._undo,    "#475569",  8),
            ("💾 Save",                self._save,    SUCCESS,   10),
        ]
        self._action_btns = []
        for txt, cmd, bg, w in btns:
            b = sbtn(ac, txt, cmd, bg=bg, w=w); b.pack(side=tk.LEFT, padx=3)
            self._action_btns.append(b)
        for b in self._action_btns[1:]: b.config(state=tk.DISABLED)

        self.status = tk.Label(f, text="", font=FS, bg=BG, fg=SUBTEXT)
        self.status.pack(anchor="w", padx=24)
        lf = tk.Frame(f, bg=BG); lf.pack(fill=tk.BOTH, expand=True, padx=24, pady=(4,20))
        _, self._log = logbox(lf)

    def _browse(self, var, kind):
        if   kind == "pcd":  p = filedialog.askopenfilename(filetypes=[("PCD","*.pcd"),("All","*.*")])
        elif kind == "save": p = filedialog.asksaveasfilename(defaultextension=".pcd",
                                                               filetypes=[("PCD","*.pcd"),("All","*.*")])
        else:                p = filedialog.askopenfilename(filetypes=[("TXT","*.txt"),("All","*.*")])
        if p:
            var.set(p)
            if kind == "pcd" and not self.save_var.get():
                base, ext = os.path.splitext(p); self.save_var.set(base + "_edited" + ext)

    def _set_status(self, msg): self.after(0, lambda: self.status.config(text=msg))

    def _load(self):
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path): messagebox.showerror("Error","Select a PCD file"); return
        _libs(); self._log(f"Loading {os.path.basename(path)} ...")
        self._pcd = o3d.io.read_point_cloud(path); self._path = path
        self._undo_stack = []; self._sel_idx = np.array([], dtype=int)
        pts = np.asarray(self._pcd.points); z = pts[:, 2]
        self.info_lbl.config(text=(
            f"{len(pts):,} pts  X[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
            f"Y[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]  Z[{z.min():.2f},{z.max():.2f}]"))
        self._log(f"  Loaded {len(pts):,} points")
        self._set_status(f"Loaded {len(pts):,} pts")
        for b in self._action_btns[1:]: b.config(state=tk.NORMAL)

    def _colorized(self):
        pts = np.asarray(self._pcd.points)
        t = (pts[:, 2] - pts[:, 2].min()) / (pts[:, 2].ptp() + 1e-9)
        colors = np.stack([np.clip(2*t-1,0,1), np.clip(2*t,0,1), np.clip(1-2*t,0,1)], 1)
        if len(self._sel_idx): colors[self._sel_idx] = [1.0, 0.1, 0.1]
        out = o3d.geometry.PointCloud(self._pcd)
        out.colors = o3d.utility.Vector3dVector(colors); return out

    def _pick(self):
        if self._pcd is None: return
        def _worker():
            self._log("Opening viewer — K=select mode, C=confirm, close window to finish")
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="EDITOR: K=select  C=confirm  then close", width=1280, height=800)
            vis.get_render_option().point_size = 2.0
            vis.get_render_option().background_color = np.array([0.05, 0.05, 0.10])
            vis.add_geometry(self._colorized()); vis.run(); vis.destroy_window()
            cropped = vis.get_cropped_geometry()
            if cropped is None or len(cropped.points) == 0:
                self._log("  No selection made."); self._set_status(f"Remaining:{len(self._pcd.points):,}"); return
            from scipy.spatial import cKDTree
            inside_pts = np.asarray(cropped.points); all_pts = np.asarray(self._pcd.points)
            tree = cKDTree(inside_pts); dists, _ = tree.query(all_pts, k=1)
            inside = np.where(dists <= 1e-4)[0]
            self._sel_idx = np.unique(np.concatenate([self._sel_idx, inside])).astype(int)
            self._log(f"  {len(inside)} pts added to selection (total:{len(self._sel_idx)})")
            self._set_status(f"Selected:{len(self._sel_idx):,}  Remaining:{len(self._pcd.points):,}")
        threading.Thread(target=_worker, daemon=True).start()

    def _auto_select(self):
        if self._pcd is None: messagebox.showerror("Error","Load a PCD first"); return
        tf = self.traj_file.get().strip()
        if not tf or not os.path.isfile(tf): messagebox.showerror("Error","Select trajectory file"); return
        _libs(); self._log("Auto-selecting trajectory band ...")
        def _worker():
            try:
                _, traj_pts = build_trajectory(tf)
            except Exception as e:
                self._log(f"  ERROR reading trajectory: {e}"); return
            tp = o3d.geometry.PointCloud()
            tp.points = o3d.utility.Vector3dVector(traj_pts.astype(np.float64))
            tree = o3d.geometry.KDTreeFlann(tp)
            pts = np.asarray(self._pcd.points); r = self.band_r.get(); r2 = r * r
            new_sel = []
            for i, pt in enumerate(pts):
                k, _, d = tree.search_radius_vector_3d(pt, r)
                if k > 0 and d[0] <= r2: new_sel.append(i)
            self._sel_idx = np.unique(
                np.concatenate([self._sel_idx, np.array(new_sel, dtype=int)])).astype(int)
            self._log(f"  Auto-selected {len(new_sel)} pts near trajectory (total:{len(self._sel_idx)})")
            self._set_status(f"Selected:{len(self._sel_idx):,}  Remaining:{len(self._pcd.points):,}")
            self._log("  → Click Preview to review (red=selected), then Delete Selected")
        threading.Thread(target=_worker, daemon=True).start()

    def _preview(self):
        if self._pcd is None: return
        pcd_col = self._colorized()
        def _show():
            o3d.visualization.draw_geometries(
                [pcd_col], window_name="Preview — red=selected for deletion",
                width=1280, height=800)
        threading.Thread(target=_show, daemon=True).start()

    def _delete(self):
        if self._pcd is None or not len(self._sel_idx):
            self._log("Nothing selected."); return
        self._undo_stack.append(o3d.geometry.PointCloud(self._pcd))
        keep = np.setdiff1d(np.arange(len(self._pcd.points)), self._sel_idx)
        self._pcd = self._pcd.select_by_index(keep.tolist())
        removed = len(self._sel_idx); self._sel_idx = np.array([], dtype=int)
        self._log(f"  Deleted {removed:,} pts. Remaining:{len(self._pcd.points):,}")
        self._set_status(f"Remaining:{len(self._pcd.points):,}")

    def _undo(self):
        if not self._undo_stack: self._log("Nothing to undo."); return
        self._pcd = self._undo_stack.pop(); self._sel_idx = np.array([], dtype=int)
        self._log(f"  Undo — restored {len(self._pcd.points):,} pts")
        self._set_status(f"Remaining:{len(self._pcd.points):,}")

    def _save(self):
        if self._pcd is None: return
        p = self.save_var.get().strip()
        if not p:
            p = filedialog.asksaveasfilename(defaultextension=".pcd",
                                              filetypes=[("PCD","*.pcd"),("All","*.*")])
        if p:
            _libs(); o3d.io.write_point_cloud(p, self._pcd)
            self._log(f"  Saved → {p}"); self._set_status(f"Saved: {os.path.basename(p)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

STRUCT_HELP = """
Keyboard shortcuts (viewer):
  F  — Floor (brown)      C  — Ceiling (grey)
  W  — Walls (coloured)   O  — Other (blue)
  D  — Door boxes (gold)  R  — Room colours
  A  — Reset (show all)   H  — This help
  Q  — Close
"""

class AnalysisTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG); self._build()

    def _build(self):
        _, f = scrollable(self)
        tk.Label(f, text="Structural Analysis & Room Detection", font=FH1, bg=BG, fg=TEXT).pack(
            anchor="w", padx=24, pady=(18,6))

        # file
        fc = card(f, padx=16, pady=12); fc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(fc, text="PCD file:", font=FB, bg=CARD, fg=TEXT).grid(row=0, column=0, sticky="w")
        self.file_var = tk.StringVar()
        tk.Entry(fc, textvariable=self.file_var, width=52, font=FB, relief="solid", bd=1).grid(
            row=0, column=1, padx=6)
        sbtn(fc, "Browse", self._browse_pcd, bg="#475569").grid(row=0, column=2)
        fc.columnconfigure(1, weight=1)
        self.traj_file = tk.StringVar()
        tk.Label(fc, text="Trajectory (opt):", font=FB, bg=CARD, fg=TEXT).grid(
            row=1, column=0, sticky="w", pady=(4,0))
        tk.Entry(fc, textvariable=self.traj_file, width=52, font=FB, relief="solid", bd=1).grid(
            row=1, column=1, padx=6, pady=(4,0))
        sbtn(fc, "Browse", lambda: self._browse_file(self.traj_file), bg="#475569").grid(
            row=1, column=2, pady=(4,0))

        # structural params
        sc = card(f, padx=16, pady=10); sc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(sc, text="Structural Detection", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self._rd=tk.DoubleVar(value=0.05);self._mp=tk.IntVar(value=16);self._mr=tk.DoubleVar(value=1.0)
        self._cpct=tk.DoubleVar(value=15);self._cmd=tk.DoubleVar(value=0.60)
        self._walo=tk.DoubleVar(value=70);self._wahi=tk.DoubleVar(value=110);self._wsp=tk.DoubleVar(value=1.2)
        lb=lambda t:tk.Label(sc,text=t,width=9,anchor="w",font=FB,bg=CARD,fg=SUBTEXT)
        lb_rd=lb("0.05 m");lb_mp=lb("16");lb_mr=lb("1.0%");lb_cpct=lb("top 15%");lb_cmd=lb("0.60 m")
        lb_walo=lb("70°");lb_wahi=lb("110°");lb_wsp=lb("1.2 m")
        slider_row(sc,1,"RANSAC dist (m):",    self._rd,  0.02,0.20,0.01,lb_rd,  lambda v:f"{float(v):.2f} m")
        slider_row(sc,2,"Max planes:",         self._mp,  4,  30,  1,   lb_mp,  lambda v:str(int(float(v))))
        slider_row(sc,3,"Min plane (% pts):",  self._mr,  0.5,10,  0.5, lb_mr,  lambda v:f"{float(v):.1f}%")
        slider_row(sc,4,"Ceil top band (%):",  self._cpct,5,  40,  1,   lb_cpct,lambda v:f"top {float(v):.0f}%")
        slider_row(sc,5,"Ceil merge dist (m):",self._cmd, 0.1,2.0, 0.1, lb_cmd, lambda v:f"{float(v):.2f} m")
        slider_row(sc,6,"Wall min angle (°):", self._walo,50, 89,  1,   lb_walo,lambda v:f"{float(v):.0f}°")
        slider_row(sc,7,"Wall max angle (°):", self._wahi,91, 130, 1,   lb_wahi,lambda v:f"{float(v):.0f}°")
        slider_row(sc,8,"Wall min Z span (m):",self._wsp, 0.5,5.0, 0.1, lb_wsp, lambda v:f"{float(v):.1f} m")

        # door params
        dc = card(f, padx=16, pady=10); dc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(dc, text="Door Detection", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self._gres=tk.DoubleVar(value=0.05);self._dmnw=tk.DoubleVar(value=0.60)
        self._dmxw=tk.DoubleVar(value=2.20);self._dmnh=tk.DoubleVar(value=1.80);self._dmxh=tk.DoubleVar(value=3.20)
        lb2=lambda t:tk.Label(dc,text=t,width=9,anchor="w",font=FB,bg=CARD,fg=SUBTEXT)
        lb_gr=lb2("0.05 m");lb_dmnw=lb2("0.60 m");lb_dmxw=lb2("2.20 m");lb_dmnh=lb2("1.80 m");lb_dmxh=lb2("3.20 m")
        slider_row(dc,1,"Wall grid res (m):",  self._gres,0.02,0.15,0.01,lb_gr,  lambda v:f"{float(v):.2f} m")
        slider_row(dc,2,"Door min width (m):", self._dmnw,0.40,1.50,0.05,lb_dmnw,lambda v:f"{float(v):.2f} m")
        slider_row(dc,3,"Door max width (m):", self._dmxw,0.80,3.00,0.05,lb_dmxw,lambda v:f"{float(v):.2f} m")
        slider_row(dc,4,"Door min height (m):",self._dmnh,1.50,2.50,0.05,lb_dmnh,lambda v:f"{float(v):.2f} m")
        slider_row(dc,5,"Door max height (m):",self._dmxh,2.00,4.00,0.05,lb_dmxh,lambda v:f"{float(v):.2f} m")

        # room params
        rc = card(f, padx=16, pady=10); rc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(rc, text="Room Detection", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        self._rres=tk.DoubleVar(value=0.10);self._rmin=tk.DoubleVar(value=2.0)
        self._rband=tk.DoubleVar(value=0.5);self._rclose=tk.IntVar(value=4)
        lb3=lambda t:tk.Label(rc,text=t,width=9,anchor="w",font=FB,bg=CARD,fg=SUBTEXT)
        lb_rr=lb3("0.10 m");lb_rm=lb3("2.0 m²");lb_rb=lb3("0.5 m");lb_rc=lb3("4")
        slider_row(rc,1,"Grid resolution (m):",  self._rres,  0.05,0.30,0.05,lb_rr,lambda v:f"{float(v):.2f} m")
        slider_row(rc,2,"Min room area (m²):",   self._rmin,  0.5, 30,  0.5, lb_rm,lambda v:f"{float(v):.1f} m²")
        slider_row(rc,3,"Traj strip radius (m):",self._rband, 0.0, 3.0, 0.1, lb_rb,lambda v:f"{float(v):.1f} m")
        slider_row(rc,4,"Wall close iterations:",self._rclose,1,   8,   1,   lb_rc,lambda v:str(int(float(v))))

        # legend
        lc = card(f, padx=16, pady=10); lc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(lc, text="Legend  (keyboard shortcuts in viewer)", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,6))
        for r, (nm, col, key) in enumerate([
            ("Floor",      "#996633", "F"),
            ("Ceiling",    "#e0e0e0", "C (all merged)"),
            ("Walls",      "#6699cc", "W (each wall diff colour)"),
            ("Other",      "#7399bf", "O"),
            ("Door boxes", "#ffd700", "D (gold wireframe)"),
            ("Rooms",      "#a855f7", "R (2D floor plan)")], 1):
            tk.Label(lc, bg=col, width=3, relief="solid").grid(row=r, column=0, padx=(0,8))
            tk.Label(lc, text=nm, font=("Segoe UI",9,"bold"), bg=CARD, fg=TEXT).grid(
                row=r, column=1, sticky="w")
            tk.Label(lc, text=f"key: {key}", font=FS, bg=CARD, fg=SUBTEXT).grid(
                row=r, column=2, sticky="w", padx=8)

        lf = tk.Frame(f, bg=BG); lf.pack(fill=tk.BOTH, expand=True, padx=24, pady=(4,0))
        _, self._log = logbox(lf)
        ac = tk.Frame(f, bg=BG); ac.pack(fill=tk.X, padx=24, pady=(8,20))
        self.run_btn = sbtn(ac, "▶  Analyse & Open Viewer", self._run, w=24)
        self.run_btn.pack(side=tk.LEFT)
        self.status = tk.Label(ac, text="", font=FS, bg=BG, fg=SUBTEXT)
        self.status.pack(side=tk.LEFT, padx=10)

    def _browse_pcd(self):
        p = filedialog.askopenfilename(filetypes=[("PCD","*.pcd"),("All","*.*")])
        if p: self.file_var.set(p)

    def _browse_file(self, var):
        p = filedialog.askopenfilename(filetypes=[("TXT","*.txt"),("All","*.*")])
        if p: var.set(p)

    def _run(self):
        path = self.file_var.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error","Select a PCD file"); return
        self.run_btn.config(state=tk.DISABLED); self.status.config(text="Loading …")
        params = dict(
            ransac_dist=self._rd.get(), max_planes=self._mp.get(),
            min_ratio=self._mr.get()/100,
            ceil_top_pct=self._cpct.get(), ceil_merge_dist=self._cmd.get(),
            wall_angle_lo=self._walo.get(), wall_angle_hi=self._wahi.get(),
            wall_min_span=self._wsp.get())

        def worker():
            try:
                _libs()
                self._log(f"Loading {os.path.basename(path)} ...")
                pcd = o3d.io.read_point_cloud(path)
                self._log(f"  {len(pcd.points):,} points")

                self._log("\n── Structural Classification ──")
                self.after(0, lambda: self.status.config(text="Detecting planes ..."))
                struct = classify_planes(pcd, log=self._log, **params)
                layers = build_struct_layers(struct)

                self._log("\n── Door Detection ──")
                self.after(0, lambda: self.status.config(text="Detecting doors ..."))
                doors = detect_doors(struct['walls'], struct['wall_models'],
                                     struct['floor_z'], struct['ceil_z'],
                                     grid_res=self._gres.get(),
                                     door_min_w=self._dmnw.get(), door_max_w=self._dmxw.get(),
                                     door_min_h=self._dmnh.get(), door_max_h=self._dmxh.get(),
                                     log=self._log)

                self._log("\n── Room Detection ──")
                self.after(0, lambda: self.status.config(text="Detecting rooms ..."))
                tf = self.traj_file.get().strip(); traj_pts = None
                if tf and os.path.isfile(tf):
                    try: _, traj_pts = build_trajectory(tf)
                    except: pass
                room_pcd, n_rooms, areas = detect_rooms(
                    pcd, traj_pts=traj_pts, traj_band=self._rband.get(),
                    grid_res=self._rres.get(), close_iter=self._rclose.get(),
                    min_room_m2=self._rmin.get(), log=self._log)

                nw = len(struct['walls']); nd = len(doors)
                hf = struct['floor'] is not None; hc = len(struct['ceiling_pcds']) > 0
                self._log(f"\n── Summary ──  floor={'✓' if hf else '✗'}  "
                          f"ceil={'✓' if hc else '✗'}  walls={nw}  doors={nd}  rooms={n_rooms}")
                self.after(0, lambda: self.status.config(
                    text=f"Floor {'✓' if hf else '✗'}  Ceil {'✓' if hc else '✗'}  "
                         f"Walls {nw}  Doors {nd}  Rooms {n_rooms}"))
                self._log(f"\nViewer open — {STRUCT_HELP}")

                vis_map = {k: True for k in ['floor','ceiling','walls','other','doors','rooms']}
                room_showing = [False]

                def _geoms():
                    G = []
                    if vis_map['rooms'] and room_showing[0]:
                        G.append(room_pcd)
                    else:
                        if vis_map['floor']   and 'floor'   in layers: G.append(layers['floor'])
                        if vis_map['ceiling'] and 'ceiling' in layers: G.append(layers['ceiling'])
                        if vis_map['walls']:   G.extend(layers.get('walls', []))
                        if vis_map['other']   and 'other'   in layers: G.append(layers['other'])
                    if vis_map['doors']:   G.extend(doors)
                    G.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
                    return G

                def _toggle(key, label, v):
                    vis_map[key] = not vis_map[key]
                    print(f"  {label}: {'ON' if vis_map[key] else 'OFF'}")
                    v.clear_geometries()
                    for g in _geoms(): v.add_geometry(g, reset_bounding_box=False)
                    v.poll_events(); v.update_renderer(); return False

                def _toggle_rooms(v):
                    room_showing[0] = not room_showing[0]
                    print(f"  Rooms: {'ON' if room_showing[0] else 'OFF'}")
                    v.clear_geometries()
                    for g in _geoms(): v.add_geometry(g, reset_bounding_box=False)
                    v.poll_events(); v.update_renderer(); return False

                def _reset_all(v):
                    for k in vis_map: vis_map[k] = True; room_showing[0] = False
                    v.clear_geometries()
                    for g in _geoms(): v.add_geometry(g, reset_bounding_box=False)
                    v.poll_events(); v.update_renderer(); return False

                vw = o3d.visualization.VisualizerWithKeyCallback()
                vw.create_window(window_name=os.path.basename(path)+" — Analysis",
                                 width=1400, height=900)
                opt = vw.get_render_option()
                opt.background_color = np.array([0.08, 0.08, 0.12]); opt.point_size = 2.0
                for g in _geoms(): vw.add_geometry(g)
                vw.register_key_callback(ord('F'), lambda v: _toggle('floor',   'Floor', v))
                vw.register_key_callback(ord('C'), lambda v: _toggle('ceiling', 'Ceiling', v))
                vw.register_key_callback(ord('W'), lambda v: _toggle('walls',   'Walls', v))
                vw.register_key_callback(ord('O'), lambda v: _toggle('other',   'Other', v))
                vw.register_key_callback(ord('D'), lambda v: _toggle('doors',   'Doors', v))
                vw.register_key_callback(ord('R'), _toggle_rooms)
                vw.register_key_callback(ord('A'), _reset_all)
                vw.register_key_callback(ord('H'), lambda v: (print(STRUCT_HELP), False)[1])
                vw.register_key_callback(ord('Q'), lambda v: (vw.close(), False)[1])
                vw.run(); vw.destroy_window()
            except Exception as e:
                import traceback; msg = str(e)
                self._log(f"ERROR: {msg}\n{traceback.format_exc()}")
                self.after(0, lambda m=msg: messagebox.showerror("Error", m))
            finally:
                self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiDAR Visualize Tool  v2.0")
        self.geometry("860x820"); self.minsize(760, 640)
        self.configure(bg=BG)

        bar = tk.Frame(self, bg=ACCENT, height=46); bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Label(bar, text="  LiDAR Visualize Tool", font=("Segoe UI",13,"bold"),
                 bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=12, pady=10)
        tk.Label(bar, text="v2.0", font=("Segoe UI",9), bg=ACCENT, fg="#bfdbfe").pack(
            side=tk.RIGHT, padx=14)

        style = ttk.Style(); style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI",10,"bold"),
                        padding=[14,8], background="#e2e8f0", foreground=SUBTEXT)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD)], foreground=[("selected", ACCENT)])
        nb = ttk.Notebook(self); nb.pack(fill=tk.BOTH, expand=True)
        nb.add(ViewerTab(nb),   text="  👁  Viewer  ")
        nb.add(EditorTab(nb),   text="  ✏  Editor  ")
        nb.add(AnalysisTab(nb), text="  🏗  Analysis  ")

        sb = tk.Frame(self, bg="#e2e8f0", height=20); sb.pack(fill=tk.X, side=tk.BOTTOM)
        sb.pack_propagate(False)
        tk.Label(sb, text="Open3D · NumPy · SciPy  |  LiDAR Visualize Tool v2.0",
                 font=("Segoe UI",7), bg="#e2e8f0", fg=SUBTEXT).pack(side=tk.RIGHT, padx=10)


def main():
    App().mainloop()

if __name__ == "__main__":
    main()
