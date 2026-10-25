"""pano.py - the Phase-4 equirectangular panorama math (T2).

PURE CPU. This module must never import vulkan, engine_vr.gl_vr,
engine_vr.passes_vr or anything under engine/ - it is the contract the
numerical gate (port/gates/gate_p4.py) probes directly, and vr.py --dry-run
imports it without touching a device.

Geometry is the BINDING spec of port/RUNBOOK.md §6:

  * lon in [-pi, pi) increases across the columns (x = 0 .. W-1),
  * lat in [ pi/2, -pi/2) decreases down the rows (y = 0 = top = +90 deg),
  * texel (x, y) has its centre at (x + 0.5, y + 0.5),
  * direction  d = (cos lat * sin lon, sin lat, cos lat * cos lon)
    in the engine's Y-up world frame (+z = north, +x = east, Runbook §6),
  * dLat = pi / H, dLon = 2 pi / W, texel angular footprint
    fp(lat) = max(dLat, dLon * cos lat)   (the wider of the two steps).

At the 2:1 spec resolutions (8192x4096 mono, 8192x2048 per eye for SBS)
dLon == dLat exactly, so fp is the constant pi/H; the latitude dependence
only appears for W < 2H inputs, which vr.py rejects (review_p4.md M1);
the helper still computes the general formula for the gate and for tests.

Shader side (T3, port/PHASE_4.md): vr.py uploads texel_footprint(w, h) as an
R32F 1 x H strip (reshaped (h, 1) at the upload seam, vr.upload_footstrip -
review_p4.md C1); the VR_PANO shader variants sample it per texel and derive
the same lon/lat from gl_FragCoord, so CPU and GPU reproduce identical rays
(for nonzero yaw/pitch/roll only since the N1 rotation fix below).

STEREO (RUNBOOK §6, brief T2): parallel projection - BOTH eyes march the
SAME direction grid; only the origins differ, +-0.032 m (half of a 0.064 m
inter-pupillary distance) along the camera-right axis.

SAME-SEED CONTRACT: the render seed is per-viewpoint, never per-eye. Both
eyes render the same uSample indices (shader gRand = f(pixel, uSample) only,
scene.frag cameraRay), the SBS post step runs post.finish once over the
stitched pair, and post.finish is called with one seed for the whole frame.
Grain, dither and every other random term therefore land on IDENTICAL pixel
coordinates in both eyes; anything else (per-eye seeds, per-eye auto
exposure) would shimmer between eyes in a headset.
"""

from __future__ import annotations

import numpy as np

IPD_M = 0.064             # inter-pupillary distance (port/RUNBOOK.md §6)
EYE_OFFSET_M = 0.5 * IPD_M  # 0.032 m per eye from the head origin


def equirect_directions(w, h, yaw=0.0, pitch=0.0, roll=0.0):
    """Equirect pixel-centre direction grid, (h, w, 3) float32, Y-up world.

    lon = -pi + (x + 0.5) * 2pi/w   (x = column)
    lat =  pi/2 - (y + 0.5) * pi/h  (row 0 = top = +90 deg)
    d   = (cos lat sin lon, sin lat, cos lat cos lon)

    yaw / pitch / roll are DEGREES applied on top of the default view
    (+z, y up) exactly the way engine Camera.basis() rotates its frame:
    the returned grid is already rotated into world space. yaw turns about
    the world up, pitch about the camera right, roll about the camera
    forward (the r/u rotation of engine/render.py:31-38 basis() copied).
    Composition matches the shader: rd_world = uCamRot * rd with uCamRot's
    columns = (r, u, f) (vr.py basis(), "rows = GLSL mat3 columns"), so the
    row-vector grid applies as dirs @ basis. review_p4.md N1: the previous
    "dirs @ basis.T" applied the INVERSE rotation, so the docstring claim
    of CPU == GPU rays was false for any nonzero angle.
    """
    w = int(w)
    h = int(h)
    if w < 2 or h < 2:
        raise ValueError(f"pano needs w,h >= 2, got {w}x{h}")
    dlon = 2.0 * np.pi / w
    dlat = np.pi / h
    lon = -np.pi + (np.arange(w, dtype=np.float64) + 0.5) * dlon
    lat = np.pi / 2.0 - (np.arange(h, dtype=np.float64) + 0.5) * dlat
    cl = np.cos(lat)[:, None]
    dirs = np.empty((h, w, 3), np.float64)
    dirs[..., 0] = cl * np.sin(lon)[None, :]
    dirs[..., 1] = np.sin(lat)[:, None]
    dirs[..., 2] = cl * np.cos(lon)[None, :]
    if yaw or pitch or roll:
        # camera basis for (yaw, pitch, roll): copied from
        # engine/render.py:31-38 Camera.basis() (f = forward,
        # r = normalize(cross(up, f)), u = cross(f, r), rows = columns of
        # the GLSL mat3); the pano default looks down +z.
        fy = np.radians(float(yaw))
        fp = np.radians(float(pitch))
        f = np.array([np.sin(fy) * np.cos(fp), np.sin(fp),
                      np.cos(fy) * np.cos(fp)], np.float64)
        r = np.cross([0.0, 1.0, 0.0], f)
        r = r / np.linalg.norm(r)
        u = np.cross(f, r)
        if roll:
            a = np.radians(float(roll))
            r, u = r * np.cos(a) + u * np.sin(a), -r * np.sin(a) + u * np.cos(a)
        basis = np.stack([r, u, f])           # rows = GLSL mat3 columns
        # review_p4.md N1: rows of basis are the COLUMNS of the shader mat3,
        # so the forward rotation is d -> R @ d, i.e. dirs @ basis; the old
        # "dirs @ basis.T" reproduced B @ d = R^T @ d (inverse rotation).
        dirs = dirs @ basis
    return dirs.astype(np.float32)


def texel_footprint(w, h):
    """Angular footprint (radians) of every texel row: (h,) float32.

    fp(lat) = max(dLat, dLon * cos(lat)) at the ROW CENTRES
    lat_y = pi/2 - (y + 0.5) * dLat (RUNBOOK §6 pixel-centre convention;
    cos >= 0 on [-pi/2, pi/2], clipped only against float rounding).
    The VR_PANO shader variants fetch this strip per texel, which replaces
    the engine's per-camera constant gPixAng = 2 uTanHalfFov / uRes.y
    (engine/shaders/scene.frag cameraRay + SCENE_PASS 98 main) with the
    exact per-ray footprint.
    """
    w = int(w)
    h = int(h)
    if w < 2 or h < 2:
        raise ValueError(f"pano needs w,h >= 2, got {w}x{h}")
    dlat = np.pi / h
    dlon = 2.0 * np.pi / w
    lat = np.pi / 2.0 - (np.arange(h, dtype=np.float64) + 0.5) * dlat
    return np.maximum(dlat, dlon * np.clip(np.cos(lat), 0.0, None)
                      ).astype(np.float32)


def stereo_pair(cam_pos, dirs, camera_right):
    """Parallel-projection stereo pair: (origin_l, origin_r, dirs).

    The equirect grid is SHARED: the returned dirs is the SAME array that
    was passed in (both eyes march identical directions - converging-fovea
    stereo is explicitly NOT used, port/RUNBOOK.md §6). The eye origins sit
    at cam_pos -+ EYE_OFFSET_M * normalize(camera_right); with the engine
    basis convention (+z forward, +x right for a north-looking camera) the
    LEFT eye is on the -right side. Returns float32 (3,) origins.
    """
    pos = np.asarray(cam_pos, np.float64).reshape(3)
    right = np.asarray(camera_right, np.float64).reshape(3)
    n = np.linalg.norm(right)
    if n == 0.0:
        raise ValueError("camera_right must be a non-zero vector")
    right = right / n
    off = EYE_OFFSET_M * right
    origin_l = (pos - off).astype(np.float32)
    origin_r = (pos + off).astype(np.float32)
    return origin_l, origin_r, dirs
