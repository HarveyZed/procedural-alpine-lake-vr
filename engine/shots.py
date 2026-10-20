"""Named still cameras (lake views + look-dev views) and helpers to place
cameras relative to the terrain."""
import math

import numpy as np

from render import Camera, Look


def axis_x(z):
    return 150.0 * math.sin(z * 0.00105 + 0.5) + 45.0 * math.sin(z * 0.0029 + 2.1)


def shore_point(world, z, side, level=0.0, inland=0.0):
    """Walk from the lake axis toward side (-1 west, +1 east) to the waterline."""
    x = axis_x(z)
    for i in range(4000):
        h = world.height(x, z)
        if h > level:
            break
        x += side * 0.5
    x += side * inland
    return x


def ground_cam(world, x, z, eye=1.7):
    return (x, max(world.height(x, z), world.water(x, z)) + eye, z)


def look_at_dir(pos, azimuth, pitch, dist=1000.0):
    a, p = math.radians(azimuth), math.radians(pitch)
    return (pos[0] + math.sin(a) * math.cos(p) * dist, pos[1] + math.sin(p) * dist, pos[2] + math.cos(a) * math.cos(p) * dist)


def stills(world):
    S = {}
    look = Look()
    # overview from high above the south end looking north up the lake
    p = (axis_x(-2600) + 200, 900.0, -2600.0)
    S["overview"] = (Camera(p, (axis_x(400), 150.0, 400.0), fov=50), look, 0.0)
    # pebble beach on the west shore looking NNE across the lake
    z = -700.0
    x = shore_point(world, z, -1, inland=2.5)
    p = ground_cam(world, x, z, 1.6)
    S["west_beach"] = (Camera(p, look_at_dir(p, 20, 4), fov=58), look, 0.0)
    # high viewpoint over the lake looking south down the valley
    p = (axis_x(1500) + 650, 520.0, 1500.0)
    S["valley_south"] = (Camera(p, (axis_x(-1200), -50.0, -1200.0), fov=60), look, 0.0)
    # meadow above the tarn looking west
    p = (-1250.0, 0.0, -300.0)
    p = ground_cam(world, p[0], p[2], 1.7)
    S["tarn_meadow"] = (Camera(p, (-1700.0, 480.0, -440.0), fov=62), look, 0.0)
    # east shore looking north-west to the head peaks
    z = 300.0
    x = shore_point(world, z, +1, inland=1.5)
    p = ground_cam(world, x, z, 1.5)
    S["east_shore"] = (Camera(p, look_at_dir(p, -25, 6), fov=60), look, 0.0)
    # sky / cloud look-dev
    p = ground_cam(world, shore_point(world, -900.0, +1, inland=3.0), -900.0, 2.0)
    S["sky"] = (Camera(p, look_at_dir(p, 10, 14), fov=70), look, 0.0)
    p = (axis_x(-400.0), 3.0, -400.0)
    S["clouds"] = (Camera(p, look_at_dir(p, 160, 22), fov=75), look, 0.0)
    # forest edge look-dev: on the east shore, looking along the treeline
    z = 200.0
    x = shore_point(world, z, +1, inland=12.0)
    p = ground_cam(world, x, z, 1.7)
    S["forest"] = (Camera(p, look_at_dir(p, 60, 8), fov=60), look, 0.0)
    x = shore_point(world, z, +1, inland=4.0)
    p = ground_cam(world, x, z, 1.7)
    S["treeline"] = (Camera(p, look_at_dir(p, 20, 4), fov=60), look, 0.0)
    # golden hour test: west shore looking north-east at the peaks in low warm light
    from dataclasses import replace as _rep
    gold = _rep(look, sun_azimuth=255.0, sun_elevation=10.0, cloud_cover=0.2)
    z = 300.0
    x = shore_point(world, z, -1, inland=2.0)
    p = ground_cam(world, x, z, 1.6)
    S["golden"] = (Camera(p, look_at_dir(p, 35, 5), fov=62), gold, 0.0)
    # waterfall + stream look-dev (needs features.json)
    f = getattr(world, "features", None)
    if f:
        wf = f["waterfall"]
        d = np.array(wf["dir"]); al = np.array([-d[1], d[0]])
        foot = np.array([wf["x"], wf["z"]])
        c = foot + d * 70.0 + al * 35.0
        p = (float(c[0]), 1.4, float(c[1]))
        S["falls"] = (Camera(p, (float(foot[0]), 16.0, float(foot[1])), fov=55), look, 3.0)
        st = f["stream"]
        a = st[2]; b = st[9]
        p = (a[0] + 6.0, max(world.height(a[0] + 6.0, a[1] - 25.0), 0.0) + 1.6, a[1] - 25.0)
        S["stream"] = (Camera(p, (b[0], b[2] + 60.0, b[1] + 400.0), fov=58), look, 3.0)
    return S
