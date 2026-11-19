"""
LiDAR Train Tool  v2.0  —  quaternion-free point cloud alignment
=================================================================
Modes (all use only PCD data, no external pose/quaternion files):

  Mode A — FGR + GICP
      Pure feature-based frame-to-frame registration.
      FPFH descriptors → Fast Global Registration → GICP fine.
      Good baseline; tolerates large frame gaps.

  Mode B — FGR + GICP + Plane-Constrained ICP
      Mode A + dominant-plane detection per frame pair.
      Plane constraints keep walls/floor aligned.
      Recommended for indoor structured environments.

  Mode C — Submap + Loop Closure
      Submap-based local registration (Mode B quality) +
      FPFH global descriptor loop detection + pose-graph optimisation.
      Best accuracy for large scenes or long corridors.

Output per run:
  <output>.pcd          — merged, voxel-downsampled, tilt-corrected map
  <output>_trajectory.txt — camera path  (x y z per frame)

Requirements:  pip install open3d numpy scipy tqdm
"""

import os, sys, glob, threading, tkinter as tk
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
#  REGISTRATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def prep(pcd, voxel, nr_mult=3, nn=30):
    """Downsample + compute oriented normals."""
    d = pcd.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * nr_mult, max_nn=nn))
    d.orient_normals_consistent_tangent_plane(nn)
    return d

def fpfh(pcd, voxel):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))

def fgr(src, tgt, sf, tf, voxel):
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        src, tgt, sf, tf,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=voxel * 1.5,
            iteration_number=128, maximum_tuple_count=2000))
    return res.transformation, res.fitness

def gicp(src, tgt, init_T, max_d, max_i):
    r = o3d.pipelines.registration.registration_generalized_icp(
        src, tgt, max_d, init_T,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, max_i))
    return r.transformation, r.fitness, r.inlier_rmse

def icp_p2plane(src, tgt, init_T, max_d, max_i):
    r = o3d.pipelines.registration.registration_icp(
        src, tgt, max_d, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, max_i))
    return r.transformation, r.fitness, r.inlier_rmse


# ═══════════════════════════════════════════════════════════════════════════════
#  PLANE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_dominant_planes(pcd, ransac_dist=0.05, max_planes=3, min_ratio=0.05):
    remaining = o3d.geometry.PointCloud(pcd)
    total = len(pcd.points); planes = []
    for _ in range(max_planes):
        if len(remaining.points) < 50: break
        model, inliers = remaining.segment_plane(ransac_dist, 3, 1000)
        if len(inliers) / total < min_ratio: break
        a, b, c, d = model; n = np.array([a, b, c]); n /= np.linalg.norm(n)
        planes.append((n, d))
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes

def augment_plane_constraints(pcd, planes, weight=50):
    if not planes: return pcd
    pts   = np.asarray(pcd.points)
    norms = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(pts)
    ctr   = pts.mean(axis=0); ep, en = [], []
    for n_vec, d in planes:
        dist = np.dot(n_vec, ctr) + d
        base = ctr - dist * n_vec
        offsets = np.random.randn(weight, 3) * 0.01
        ep.append(base + offsets); en.append(np.tile(n_vec, (weight, 1)))
    out = o3d.geometry.PointCloud()
    out.points  = o3d.utility.Vector3dVector(np.vstack([pts, *ep]))
    out.normals = o3d.utility.Vector3dVector(np.vstack([norms, *en]))
    return out

def icp_plane_constrained(src, tgt, sp, tp, init_T, max_d, max_i, weight):
    sa = augment_plane_constraints(src, sp, weight)
    ta = augment_plane_constraints(tgt, tp, weight)
    if not sa.has_normals(): sa.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    if not ta.has_normals(): ta.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    r = o3d.pipelines.registration.registration_icp(
        sa, ta, max_d, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(1e-7, 1e-7, max_i))
    return r.transformation, r.fitness, r.inlier_rmse

def plane_tilt_correction(merged, log):
    """Detect dominant horizontal plane and return a levelling transform."""
    total = len(merged.points); remaining = o3d.geometry.PointCloud(merged)
    best_n, best_cnt = None, 0
    for k in range(10):
        if len(remaining.points) < 50: break
        model, inliers = remaining.segment_plane(0.06, 3, 3000)
        a, b, c, d = model; n = np.array([a, b, c]); n /= np.linalg.norm(n)
        z_al = abs(n[2]); ratio = len(inliers) / total
        log(f"  Plane {k+1}: {len(inliers):,} pts ({ratio*100:.1f}%)  z-align={z_al:.3f}")
        if z_al > 0.85 and ratio > 0.08 and len(inliers) > best_cnt:
            best_n = n if n[2] > 0 else -n; best_cnt = len(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
    if best_n is None: log("  No horizontal plane found — skipping tilt correction."); return None
    tilt = np.degrees(np.arccos(np.clip(best_n[2], -1, 1)))
    log(f"  Tilt: {tilt:.4f}°")
    if tilt < 0.05: log("  Negligible tilt — skipping."); return None
    z = np.array([0., 0., 1.]); ax = np.cross(best_n, z); ax /= np.linalg.norm(ax)
    angle = np.arccos(np.clip(np.dot(best_n, z), -1, 1))
    R = Rotation.from_rotvec(ax * angle).as_matrix()
    T = np.eye(4); T[:3, :3] = R; return T

def merge_save(frames, corr_T, voxel, out_path, log, poses=None):
    log(f"Merging {len(frames)} frames ...")
    merged = o3d.geometry.PointCloud()
    for pc in frames: merged += pc
    log(f"  Pre-downsample:  {len(merged.points):,}")
    merged = merged.voxel_down_sample(voxel)
    log(f"  Post-downsample: {len(merged.points):,}")
    if corr_T is not None:
        log("  Applying tilt correction ...")
        merged.transform(corr_T)
    o3d.io.write_point_cloud(out_path, merged)
    log(f"Saved → {out_path}")
    if poses is not None:
        traj = np.array([T[:3, 3] for T in poses])
        if corr_T is not None:
            traj = (corr_T[:3, :3] @ traj.T).T
        tp = os.path.splitext(out_path)[0] + "_trajectory.txt"
        np.savetxt(tp, traj, fmt="%.6f", header="x y z")
        log(f"Trajectory → {tp}")
        return tp
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  LOOP CLOSURE  (Mode C)
# ═══════════════════════════════════════════════════════════════════════════════

def global_descriptor(pcd, voxel):
    arr = np.array(fpfh(pcd, voxel).data).T
    return arr.mean(axis=0)

def detect_loop(cur_desc, hist_descs, hist_poses, cur_pose, min_dist=3.0, top_k=3):
    if not hist_descs: return []
    cur_xyz = cur_pose[:3, 3]; cands = []
    for i, (desc, pose) in enumerate(zip(hist_descs, hist_poses)):
        if np.linalg.norm(cur_xyz - pose[:3, 3]) < min_dist: continue
        sim = np.dot(cur_desc, desc) / (np.linalg.norm(cur_desc) * np.linalg.norm(desc) + 1e-9)
        cands.append((sim, i))
    cands.sort(reverse=True)
    return [i for _, i in cands[:top_k]]

def pose_graph_optimize(poses, loop_edges, log):
    if not loop_edges: log("  No loop edges — skipping optimisation"); return poses
    corrected = [p.copy() for p in poses]; N = len(corrected)
    for ei, (i, j, T_ij) in enumerate(loop_edges):
        if i >= N or j >= N or i >= j: continue
        T_i = corrected[i]; T_j = corrected[j]
        T_err = T_ij @ np.linalg.inv(np.linalg.inv(T_i) @ T_j)
        R_err = T_err[:3, :3]; t_err = T_err[:3, 3]
        rv = Rotation.from_matrix(R_err).as_rotvec(); path = j - i
        for k in range(1, path + 1):
            a = k / path; dT = np.eye(4)
            dT[:3, :3] = Rotation.from_rotvec(rv * a).as_matrix()
            dT[:3, 3]  = t_err * a
            corrected[i + k] = corrected[i + k] @ dT
        log(f"  Loop {ei+1}: {i}↔{j}  Δt={np.linalg.norm(t_err):.4f}m")
    return corrected


# ═══════════════════════════════════════════════════════════════════════════════
#  ALIGNMENT MODES  (no quaternion)
# ═══════════════════════════════════════════════════════════════════════════════

def _fgr_init(cur_c, prev_c, cur_fpfh, prev_fpfh, vc, fgr_thresh, log, i):
    """Try FGR; fall back to identity if below threshold."""
    T_fgr, fit = fgr(cur_c, prev_c, cur_fpfh, prev_fpfh, vc)
    if fit >= fgr_thresh:
        return T_fgr, fit, True
    # FGR failed — use identity (assume slow motion between consecutive frames)
    log(f"  ⚠ frame {i:04d}  FGR low ({fit:.3f}) → identity init")
    return np.eye(4), fit, False


# ── Mode A: FGR + GICP ───────────────────────────────────────────────────────
def process_mode_a(data_dir, out_path, params, log, prog):
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "pointcloud_*.pcd")))
    N = len(pcd_files)
    if N == 0: log("ERROR: no pointcloud_*.pcd files found"); return None
    log(f"Mode A: FGR + GICP  |  {N} frames")
    vc = params["voxel_c"]; vf = params["voxel_f"]
    d2 = params["d2"];      i2 = params["i2"]
    fgr_thresh = params.get("fgr_thresh", 0.25)

    raw0  = o3d.io.read_point_cloud(pcd_files[0])
    prev_c = prep(raw0, vc); prev_f = prep(raw0, vf)
    prev_fpfh = fpfh(prev_c, vc)
    frames = [o3d.geometry.PointCloud(prev_f)]; poses = [np.eye(4)]
    prev_pose = np.eye(4); low = 0; prog(1, N)

    for i in range(1, N):
        raw = o3d.io.read_point_cloud(pcd_files[i])
        if not len(raw.points): prog(i + 1, N); continue
        cur_c = prep(raw, vc); cur_f = prep(raw, vf)
        cur_fpfh_feat = fpfh(cur_c, vc)

        init_T, fit_fgr, fgr_ok = _fgr_init(cur_c, prev_c, cur_fpfh_feat, prev_fpfh, vc, fgr_thresh, log, i)
        T2, f2, r2 = gicp(cur_c, prev_c, init_T, d2, i2)
        final_T = T2 if f2 >= 0.20 else init_T
        if f2 < 0.20: low += 1; log(f"  ⚠ frame {i:04d}  GICP low ({f2:.3f})")

        cur_pose = prev_pose @ final_T
        pc_w = o3d.geometry.PointCloud(cur_f); pc_w.transform(cur_pose)
        frames.append(pc_w); poses.append(cur_pose)
        prev_c, prev_f, prev_fpfh, prev_pose = cur_c, cur_f, cur_fpfh_feat, cur_pose

        if i % 50 == 0: log(f"  frame {i:04d}  FGR={fit_fgr:.3f}  GICP={f2:.4f}  rmse={r2:.4f}")
        prog(i + 1, N)

    log(f"Low-fitness: {low}/{N}")
    preview = o3d.geometry.PointCloud()
    for pc in frames: preview += pc
    corr_T = plane_tilt_correction(preview.voxel_down_sample(vf * 3), log)
    return merge_save(frames, corr_T, vf, out_path, log, poses)


# ── Mode B: FGR + GICP + Plane-Constrained ICP ───────────────────────────────
def process_mode_b(data_dir, out_path, params, log, prog):
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "pointcloud_*.pcd")))
    N = len(pcd_files)
    if N == 0: log("ERROR: no pointcloud_*.pcd files found"); return None
    log(f"Mode B: FGR + GICP + Plane-Constrained ICP  |  {N} frames")
    vc = params["voxel_c"]; vf = params["voxel_f"]
    d2 = params["d2"];      d3 = params["d3"]
    i2 = params["i2"];      i3 = params["i3"]
    fgr_thresh  = params.get("fgr_thresh",  0.25)
    use_pc      = params.get("use_plane_constr", True)
    pr = params.get("plane_ransac", 0.06); pm = params.get("plane_max", 3); pw = params.get("plane_weight", 80)

    raw0   = o3d.io.read_point_cloud(pcd_files[0])
    prev_c = prep(raw0, vc); prev_f = prep(raw0, vf)
    prev_fpfh = fpfh(prev_c, vc)
    prev_planes = detect_dominant_planes(prev_f, pr, pm) if use_pc else []
    frames = [o3d.geometry.PointCloud(prev_f)]; poses = [np.eye(4)]
    prev_pose = np.eye(4); low = 0; prog(1, N)

    for i in range(1, N):
        raw = o3d.io.read_point_cloud(pcd_files[i])
        if not len(raw.points): prog(i + 1, N); continue
        cur_c = prep(raw, vc); cur_f = prep(raw, vf)
        cur_fpfh_feat = fpfh(cur_c, vc)
        cur_planes = detect_dominant_planes(cur_f, pr, pm) if use_pc else []

        init_T, fit_fgr, _ = _fgr_init(cur_c, prev_c, cur_fpfh_feat, prev_fpfh, vc, fgr_thresh, log, i)
        T2, f2, _ = gicp(cur_c, prev_c, init_T, d2, i2)

        if use_pc and (cur_planes or prev_planes):
            T3, f3, r3 = icp_plane_constrained(cur_f, prev_f, cur_planes, prev_planes, T2, d3, i3, pw)
        else:
            T3, f3, r3 = icp_p2plane(cur_f, prev_f, T2, d3, i3)

        if   f3 >= 0.35: final_T = T3
        elif f2 >= 0.20: final_T = T2; (i % 20 == 0 and log(f"  ⚠ frame {i:04d}  cICP={f3:.3f}"))
        else:            final_T = init_T; low += 1; log(f"  ⚠ frame {i:04d}  all low")

        cur_pose = prev_pose @ final_T
        pc_w = o3d.geometry.PointCloud(cur_f); pc_w.transform(cur_pose)
        frames.append(pc_w); poses.append(cur_pose)
        prev_c, prev_f, prev_fpfh, prev_planes, prev_pose = cur_c, cur_f, cur_fpfh_feat, cur_planes, cur_pose

        if i % 50 == 0: log(f"  frame {i:04d}  FGR={fit_fgr:.3f}  GICP={f2:.3f}  cICP={f3:.4f}")
        prog(i + 1, N)

    log(f"Low-fitness (all stages): {low}/{N}")
    preview = o3d.geometry.PointCloud()
    for pc in frames: preview += pc
    corr_T = plane_tilt_correction(preview.voxel_down_sample(vf * 3), log)
    return merge_save(frames, corr_T, vf, out_path, log, poses)


# ── Mode C: Submap + Loop Closure ─────────────────────────────────────────────
def process_mode_c(data_dir, out_path, params, log, prog):
    pcd_files = sorted(glob.glob(os.path.join(data_dir, "pointcloud_*.pcd")))
    N = len(pcd_files)
    if N == 0: log("ERROR: no pointcloud_*.pcd files found"); return None
    log(f"Mode C: Submap + Loop Closure  |  {N} frames")
    vc = params["voxel_c"]; vf = params["voxel_f"]
    d2 = params["d2"];      d3 = params["d3"]
    i2 = params["i2"];      i3 = params["i3"]
    fgr_thresh    = params.get("fgr_thresh",    0.25)
    use_pc        = params.get("use_plane_constr", True)
    pr  = params.get("plane_ransac",   0.03)
    pm  = params.get("plane_max",       5)
    pw  = params.get("plane_weight",  250)
    sub_k         = params.get("submap_k",         5)
    loop_interval = params.get("loop_interval",   30)
    loop_min_dist = params.get("loop_min_dist",  3.0)
    loop_gicp_d   = params.get("loop_gicp_dist", 0.5)
    loop_fit_th   = params.get("loop_fit_thresh",0.35)

    raw0   = o3d.io.read_point_cloud(pcd_files[0])
    ref_c  = prep(raw0, vc); ref_f = prep(raw0, vf)
    all_c  = [ref_c]; all_f = [ref_f]
    all_poses = [np.eye(4)]; rel_history = [np.eye(4)]
    all_descs = [global_descriptor(ref_c, vc)]
    loop_edges = []; loop_history = []
    prev_fpfh = fpfh(ref_c, vc); prev_pose = np.eye(4)
    prog(1, N)

    for i in range(1, N):
        raw = o3d.io.read_point_cloud(pcd_files[i])
        if not len(raw.points):
            all_c.append(o3d.geometry.PointCloud()); all_f.append(o3d.geometry.PointCloud())
            all_poses.append(all_poses[-1]); rel_history.append(np.eye(4))
            all_descs.append(all_descs[-1]); prog(i + 1, N); continue

        cur_c = prep(raw, vc); cur_f = prep(raw, vf)
        cur_fpfh_feat = fpfh(cur_c, vc)

        # ── initial relative transform via FGR ──
        init_rel, fit_fgr, _ = _fgr_init(cur_c, all_c[-1], cur_fpfh_feat, prev_fpfh, vc, fgr_thresh, log, i)

        # ── build local submap in frame i-1 coords ──
        sub_indices = list(range(max(0, i - sub_k), i))
        submap_c = o3d.geometry.PointCloud(); submap_f = o3d.geometry.PointCloud()
        cum_T = np.eye(4); added = 0
        for step, j in enumerate(reversed(sub_indices)):
            if step > 0:
                r_idx = i - step
                if 0 < r_idx < len(rel_history):
                    cum_T = cum_T @ np.linalg.inv(rel_history[r_idx])
            if len(all_c[j].points) == 0: continue
            pc = o3d.geometry.PointCloud(all_c[j]); pc.transform(cum_T); submap_c += pc
            pf = o3d.geometry.PointCloud(all_f[j]); pf.transform(cum_T); submap_f += pf
            added += 1

        if added == 0 or len(submap_c.points) < 50:
            submap_c = o3d.geometry.PointCloud(all_c[i - 1])
            submap_f = o3d.geometry.PointCloud(all_f[i - 1])

        submap_c = submap_c.voxel_down_sample(vc)
        submap_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vc * 3, max_nn=30))
        submap_f = submap_f.voxel_down_sample(vf)
        submap_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vf * 3, max_nn=30))

        # ── GICP against submap ──
        T_gicp, f_gicp, _ = gicp(cur_c, submap_c, init_rel, d2, i2)

        # ── plane-constrained fine ICP ──
        if use_pc:
            cur_f_tmp = o3d.geometry.PointCloud(cur_f); cur_f_tmp.transform(T_gicp)
            cp = detect_dominant_planes(cur_f_tmp, pr, pm)
            sp = detect_dominant_planes(submap_f,  pr, pm)
            T_fine, f_fine, r_fine = icp_plane_constrained(cur_f_tmp, submap_f, cp, sp, np.eye(4), d3, i3, pw)
            T_refined = T_fine @ T_gicp
        else:
            T_fine, f_fine, r_fine = icp_p2plane(cur_f, submap_f, T_gicp, d3, i3)
            T_refined = T_fine

        if   f_fine  >= 0.30: final_rel = T_refined
        elif f_gicp  >= 0.20: final_rel = T_gicp; (i % 20 == 0 and log(f"  ⚠ frame {i:04d}  fine low ({f_fine:.3f})"))
        else:                 final_rel = init_rel; log(f"  ⚠ frame {i:04d}  all low, using FGR init")

        cur_pose = prev_pose @ final_rel
        all_c.append(cur_c); all_f.append(cur_f)
        all_poses.append(cur_pose); rel_history.append(final_rel)
        all_descs.append(global_descriptor(cur_c, vc))

        # ── loop closure ──
        if i % loop_interval == 0 and i >= loop_interval * 2:
            hist_d = [all_descs[k] for k in loop_history]
            hist_p = [all_poses[k] for k in loop_history]
            cands  = detect_loop(all_descs[i], hist_d, hist_p, cur_pose, loop_min_dist)
            for ci in cands[:1]:
                abs_ci = loop_history[ci]
                ref_sub = o3d.geometry.PointCloud()
                for si in range(max(0, abs_ci - 2), min(len(all_c), abs_ci + 3)):
                    if not len(all_c[si].points): continue
                    tmp = o3d.geometry.PointCloud(all_c[si]); tmp.transform(all_poses[si]); ref_sub += tmp
                ref_sub = ref_sub.voxel_down_sample(vc)
                ref_sub.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vc * 3, max_nn=30))
                cw = o3d.geometry.PointCloud(cur_c); cw.transform(cur_pose)
                T_lp, f_lp, _ = gicp(cw, ref_sub, np.eye(4), loop_gicp_d, 100)
                if f_lp >= loop_fit_th:
                    loop_edges.append((abs_ci, i, np.linalg.inv(all_poses[abs_ci]) @ (T_lp @ cur_pose)))
                    log(f"  ✓ Loop: {abs_ci}↔{i}  fit={f_lp:.3f}"); break
            loop_history.append(i)

        prev_fpfh = cur_fpfh_feat; prev_pose = cur_pose
        if i % 50 == 0: log(f"  frame {i:04d}  FGR={fit_fgr:.3f}  GICP={f_gicp:.3f}  fine={f_fine:.4f}  sub={added}fr")
        prog(i + 1, N)

    opt_poses = pose_graph_optimize(all_poses, loop_edges, log) if loop_edges else all_poses
    frames_w = []
    for i in range(N):
        if not len(all_f[i].points): continue
        tmp = o3d.geometry.PointCloud(all_f[i]); tmp.transform(opt_poses[i]); frames_w.append(tmp)

    log("Global tilt correction ...")
    preview = o3d.geometry.PointCloud()
    for pc in frames_w: preview += pc
    corr_T = plane_tilt_correction(preview.voxel_down_sample(vf * 3), log)
    return merge_save(frames_w, corr_T, vf, out_path, log, opt_poses)


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
    txt = tk.Text(parent, height=9, font=FM, bg="#0f172a", fg="#94a3b8",
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
#  MAIN PROCESS TAB
# ═══════════════════════════════════════════════════════════════════════════════

class TrainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LiDAR Train Tool  v2.0")
        self.geometry("900x860"); self.minsize(760, 680)
        self.configure(bg=BG)

        # ── title bar ──
        bar = tk.Frame(self, bg=ACCENT, height=46); bar.pack(fill=tk.X); bar.pack_propagate(False)
        tk.Label(bar, text="  LiDAR Train Tool", font=("Segoe UI",13,"bold"),
                 bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=12, pady=10)
        tk.Label(bar, text="v2.0 — quaternion-free", font=("Segoe UI",9),
                 bg=ACCENT, fg="#bfdbfe").pack(side=tk.RIGHT, padx=14)

        _, f = scrollable(self)

        # ── input card ──
        ic = card(f, padx=16, pady=12); ic.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(ic, text="Input", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,8))
        self.dir_var = tk.StringVar(); self.out_var = tk.StringVar(value="aligned_map.pcd")
        for r, (lbl, var, kind) in enumerate([
                ("PCD folder:",  self.dir_var, "dir"),
                ("Output file:", self.out_var, "save")], 1):
            tk.Label(ic, text=lbl, font=FB, bg=CARD, fg=TEXT).grid(row=r, column=0, sticky="w", pady=3)
            tk.Entry(ic, textvariable=var, width=52, font=FB, relief="solid", bd=1).grid(
                row=r, column=1, padx=6)
            sbtn(ic, "Browse", (lambda k=kind, v=var: self._browse(v, k)),
                 bg="#475569").grid(row=r, column=2)
        ic.columnconfigure(1, weight=1)

        # ── mode card ──
        mc = card(f, padx=16, pady=12); mc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(mc, text="Alignment Mode", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,8))
        self.mode_var = tk.StringVar(value="B")
        for col, (val, title, tip) in enumerate([
            ("A", "FGR + GICP",
             "Feature matching + GICP.\nFast, good for simple paths."),
            ("B", "FGR + GICP + P-ICP",
             "+ plane constraints.\nRecommended for indoor scenes."),
            ("C", "Submap + Loop Closure",
             "Submap GICP + loop detection\n+ pose-graph optimisation.\nBest accuracy."),
        ]):
            frm = tk.Frame(mc, bg=CARD); frm.grid(row=1, column=col, sticky="nw", padx=(0,16))
            tk.Radiobutton(frm, text=f"{val} — {title}", variable=self.mode_var, value=val,
                           font=("Segoe UI",9,"bold"), bg=CARD, fg=TEXT,
                           activebackground=CARD, selectcolor="#dbeafe").pack(anchor="w")
            tk.Label(frm, text=tip, font=FS, bg=CARD, fg=SUBTEXT, justify=tk.LEFT).pack(anchor="w", padx=18)

        # ── params card ──
        pc = card(f, padx=16, pady=12); pc.pack(fill=tk.X, padx=24, pady=4)
        tk.Label(pc, text="Parameters", font=FH2, bg=CARD, fg=TEXT).grid(
            row=0, columnspan=3, sticky="w", pady=(0,8))
        def plbl(t): return tk.Label(pc, text=t, width=10, anchor="w", font=FB, bg=CARD, fg=SUBTEXT)

        self._vc  = tk.DoubleVar(value=0.08); self._vf  = tk.DoubleVar(value=0.03)
        self._d2  = tk.DoubleVar(value=0.50); self._d3  = tk.DoubleVar(value=0.15)
        self._i2  = tk.IntVar(value=100);     self._i3  = tk.IntVar(value=200)
        self._fgr = tk.DoubleVar(value=0.25)
        self._use_pc = tk.BooleanVar(value=True)
        self._pr  = tk.DoubleVar(value=0.03); self._pm  = tk.IntVar(value=5);  self._pw  = tk.IntVar(value=250)
        self._sk  = tk.IntVar(value=5);       self._li  = tk.IntVar(value=30)
        self._lmd = tk.DoubleVar(value=3.0);  self._lft = tk.DoubleVar(value=0.35)

        lb_vc=plbl("0.08 m");lb_vf=plbl("0.03 m");lb_d2=plbl("0.50 m");lb_d3=plbl("0.15 m")
        lb_i2=plbl("100");lb_i3=plbl("200");lb_fgr=plbl("0.25")
        lb_pr=plbl("0.03 m");lb_pm=plbl("5");lb_pw=plbl("250 pts")
        lb_sk=plbl("5 fr");lb_li=plbl("30 fr");lb_lmd=plbl("3.0 m");lb_lft=plbl("0.35")

        tk.Label(pc, text="── Shared (A/B/C) ──", font=FS, bg=CARD, fg=SUBTEXT).grid(
            row=1, columnspan=3, sticky="w")
        slider_row(pc,  2, "Coarse voxel (m):",  self._vc,  0.04, 0.20, 0.01, lb_vc,  lambda v: f"{float(v):.2f} m")
        slider_row(pc,  3, "Fine voxel (m):",    self._vf,  0.01, 0.08, 0.01, lb_vf,  lambda v: f"{float(v):.2f} m")
        slider_row(pc,  4, "GICP dist (m):",     self._d2,  0.05, 1.0,  0.05, lb_d2,  lambda v: f"{float(v):.2f} m")
        slider_row(pc,  5, "GICP iter:",         self._i2,  50,   300,  10,   lb_i2,  lambda v: str(int(float(v))))
        slider_row(pc,  6, "FGR min fitness:",   self._fgr, 0.05, 0.9,  0.05, lb_fgr, lambda v: f"{float(v):.2f}")
        sep(pc, 7)
        tk.Label(pc, text="── Mode B / C ──", font=FS, bg=CARD, fg=SUBTEXT).grid(
            row=8, columnspan=3, sticky="w")
        slider_row(pc,  9, "Fine ICP dist (m):", self._d3,  0.05, 0.5,  0.05, lb_d3,  lambda v: f"{float(v):.2f} m")
        slider_row(pc, 10, "Fine ICP iter:",     self._i3,  50,   400,  10,   lb_i3,  lambda v: str(int(float(v))))
        tk.Checkbutton(pc, text="Plane constraints", variable=self._use_pc,
                       font=FB, bg=CARD, fg=TEXT,
                       activebackground=CARD, selectcolor="#dbeafe").grid(
            row=11, columnspan=3, sticky="w")
        slider_row(pc, 12, "Plane RANSAC (m):",  self._pr,  0.01, 0.15, 0.01, lb_pr, lambda v: f"{float(v):.2f} m")
        slider_row(pc, 13, "Max planes:",        self._pm,  1,    8,    1,    lb_pm, lambda v: str(int(float(v))))
        slider_row(pc, 14, "Constraint weight:", self._pw, 10,   400,  10,   lb_pw, lambda v: f"{int(float(v))} pts")
        sep(pc, 15)
        tk.Label(pc, text="── Mode C only ──", font=FS, bg=CARD, fg=SUBTEXT).grid(
            row=16, columnspan=3, sticky="w")
        slider_row(pc, 17, "Submap K (frames):",   self._sk,  2,  15,  1,   lb_sk,  lambda v: f"{int(float(v))} fr")
        slider_row(pc, 18, "Loop interval:",        self._li,  10, 100, 5,   lb_li,  lambda v: f"{int(float(v))} fr")
        slider_row(pc, 19, "Loop min dist (m):",    self._lmd, 1.0,10,  0.5, lb_lmd, lambda v: f"{float(v):.1f} m")
        slider_row(pc, 20, "Loop ICP fitness:",     self._lft, 0.1,0.8, 0.05,lb_lft, lambda v: f"{float(v):.2f}")

        # ── run bar ──
        bot = tk.Frame(f, bg=BG); bot.pack(fill=tk.X, padx=24, pady=8)
        self.run_btn = sbtn(bot, "▶  Start", self._start, w=16); self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = sbtn(bot, "■  Stop", self._stop, bg=DANGER, w=8)
        self.stop_btn.pack(side=tk.LEFT, padx=6); self.stop_btn.config(state=tk.DISABLED)
        self.prog_var = tk.DoubleVar()
        ttk.Progressbar(bot, variable=self.prog_var, maximum=100, length=260).pack(side=tk.LEFT, padx=8)
        self.prog_lbl = tk.Label(bot, text="", font=FS, bg=BG, fg=SUBTEXT)
        self.prog_lbl.pack(side=tk.LEFT)

        lf = tk.Frame(f, bg=BG); lf.pack(fill=tk.BOTH, expand=True, padx=24, pady=(2,20))
        _, self._log = logbox(lf)

        # ── status bar ──
        sb2 = tk.Frame(self, bg="#e2e8f0", height=20); sb2.pack(fill=tk.X, side=tk.BOTTOM)
        sb2.pack_propagate(False)
        tk.Label(sb2, text="Open3D · NumPy · SciPy  |  LiDAR Train Tool v2.0 — no quaternion required",
                 font=("Segoe UI",7), bg="#e2e8f0", fg=SUBTEXT).pack(side=tk.RIGHT, padx=10)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _browse(self, var, kind):
        if   kind == "dir":  p = filedialog.askdirectory()
        elif kind == "save": p = filedialog.asksaveasfilename(
            defaultextension=".pcd", filetypes=[("PCD","*.pcd"),("All","*.*")])
        else:                p = filedialog.askopenfilename(
            filetypes=[("PCD","*.pcd"),("All","*.*")])
        if p: var.set(p)

    def _prog(self, i, N):
        def _do():
            self.prog_var.set(100 * i / max(N, 1))
            self.prog_lbl.config(text=f"{i}/{N}")
        self.after(0, _do)

    def _params(self):
        return {
            "voxel_c": self._vc.get(),  "voxel_f":  self._vf.get(),
            "d2":      self._d2.get(),  "d3":       self._d3.get(),
            "i2":      self._i2.get(),  "i3":       self._i3.get(),
            "fgr_thresh":        self._fgr.get(),
            "use_plane_constr":  self._use_pc.get(),
            "plane_ransac":      self._pr.get(),
            "plane_max":         self._pm.get(),
            "plane_weight":      self._pw.get(),
            "submap_k":          self._sk.get(),
            "loop_interval":     self._li.get(),
            "loop_min_dist":     self._lmd.get(),
            "loop_fit_thresh":   self._lft.get(),
            "loop_gicp_dist":    self._d2.get(),
        }

    def _start(self):
        dd = self.dir_var.get().strip(); op = self.out_var.get().strip()
        if not os.path.isdir(dd):  messagebox.showerror("Error", "Invalid PCD folder"); return
        if not op:                 messagebox.showerror("Error", "Specify output path"); return
        self.run_btn.config(state=tk.DISABLED); self.stop_btn.config(state=tk.NORMAL)
        self.prog_var.set(0); self._log("=== Starting ===")
        mode = self.mode_var.get()

        def worker():
            try:
                _libs()
                p = self._params()
                if   mode == "A": process_mode_a(dd, op, p, self._log, self._prog)
                elif mode == "B": process_mode_b(dd, op, p, self._log, self._prog)
                elif mode == "C": process_mode_c(dd, op, p, self._log, self._prog)
                self._log("=== Done! ===")
                self.after(0, lambda: messagebox.showinfo("Done", "Alignment complete!"))
            except Exception as e:
                import traceback; self._log(f"ERROR: {e}\n{traceback.format_exc()}")
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
                self.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))
        threading.Thread(target=worker, daemon=True).start()

    def _stop(self):
        self._log("--- Stop requested (current frame will finish) ---")


def main():
    TrainApp().mainloop()

if __name__ == "__main__":
    main()
