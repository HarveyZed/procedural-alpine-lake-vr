#!/usr/bin/env python3
"""P0 Spike B — wgpu-py (wgpu 0.32.0, native Metal backend via bundled wgpu-native).

Same LOCKED scene + fill pattern as spike/p0_a (see shaders/sphere.frag header).
No external toolchain: WGSL is compiled by wgpu's naga inside libwgpu_native.
256x256 single frame. Outputs: sphere.ppm (P6), buf_check.txt, WALL_SECONDS line.
"""
import time, pathlib, struct
import wgpu
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
W = H = 256
COUNT = 1024
t0 = time.perf_counter()

FILL_WGSL = """
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(16, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i: u32 = gid.x;
    if (i < 1024u) {
        data[i] = f32((i * i) % 997u);
    }
}
"""

SCENE_WGSL = """
// Fullscreen triangle, raymarched analytic sphere — mirrors spike/p0_a/shaders/{fs.vert,sphere.frag}.
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var p = array(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let ix: u32 = u32(pos.x);
    let iy: u32 = u32(pos.y);
    let dx: f32 = (f32(ix) + 0.5) / 256.0 * 2.0 - 1.0;
    let dy: f32 = 1.0 - (f32(iy) + 0.5) / 256.0 * 2.0;
    let d: vec3<f32> = normalize(vec3<f32>(dx, dy, 2.0));
    let C: vec3<f32> = vec3<f32>(0.0, 0.0, 1.2);
    let r: f32 = 0.5;
    let L: vec3<f32> = normalize(vec3<f32>(-0.5, 0.7, -0.6));

    var t: f32 = 0.01;
    var hit: f32 = -1.0;
    for (var i: i32 = 0; i < 128; i = i + 1) {
        let p: vec3<f32> = d * t - C;
        let sd: f32 = length(p) - r;
        if (sd < 0.0001) { hit = t; break; }
        t = t + max(sd, 0.005);
        if (t > 10.0) { break; }
    }

    var c: vec3<f32>;
    if (hit > 0.0) {
        let n: vec3<f32> = normalize(d * hit - C);
        let ndl: f32 = max(dot(n, L), 0.0);
        c = vec3<f32>(0.80, 0.30, 0.25) * (0.18 + 0.82 * ndl);
    } else {
        let g: f32 = 0.5 * d.y + 0.5;
        c = mix(vec3<f32>(0.25, 0.40, 0.60), vec3<f32>(0.65, 0.80, 0.95), g);
    }
    c = pow(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(c, 1.0);
}
"""

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
assert adapter is not None, "no wgpu adapter"
info = adapter.info
device = adapter.request_device_sync(label="p0b")

# ---------- compute: fill pattern ----------
fill_buf = device.create_buffer(size=COUNT * 4,
    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
read_buf = device.create_buffer(size=COUNT * 4,
    usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
bgl = device.create_bind_group_layout(entries=[dict(binding=0, visibility=wgpu.ShaderStage.COMPUTE,
    buffer=dict(type=wgpu.BufferBindingType.storage))])
bgl_fill = device.create_bind_group(layout=bgl, entries=[dict(binding=0, resource=fill_buf)])
pl = device.create_pipeline_layout(bind_group_layouts=[bgl])
cpipe = device.create_compute_pipeline(layout=pl,
    compute=dict(module=device.create_shader_module(label="fill", code=FILL_WGSL), entry_point="main"))

# ---------- render: sphere raymarch ----------
tex = device.create_texture(size=(W, H, 1), sample_count=1, mip_level_count=1,
    format=wgpu.TextureFormat.rgba8unorm, dimension="2d",
    usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
px_buf = device.create_buffer(size=W * H * 4, usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
rpipe = device.create_render_pipeline(
    layout=device.create_pipeline_layout(bind_group_layouts=[]),
    vertex=dict(module=device.create_shader_module(label="scene", code=SCENE_WGSL), entry_point="vs"),
    primitive=dict(topology="triangle-list", cull_mode=None),
    fragment=dict(module=device.create_shader_module(label="scene", code=SCENE_WGSL),
                  entry_point="fs", targets=[dict(format=wgpu.TextureFormat.rgba8unorm)]),
    multisample=dict(count=1))

enc = device.create_command_encoder()
cp = enc.begin_compute_pass(label="fill")
cp.set_pipeline(cpipe)
cp.set_bind_group(0, bgl_fill)
cp.dispatch_workgroups(COUNT // 16, 1, 1)
cp.end()
enc.copy_buffer_to_buffer(fill_buf, 0, read_buf, 0, COUNT * 4)
rp = enc.begin_render_pass(color_attachments=[dict(
    view=tex.create_view(), clear_value=(0.0, 0.0, 0.0, 1.0), load_op="clear", store_op="store")],
    depth_stencil_attachment=None)
rp.set_pipeline(rpipe)
rp.draw(3)  # wgpu-py apidiff: first arg is vertex_count (NOT WebGPU IDL order)
rp.end()
enc.copy_texture_to_buffer(dict(texture=tex), dict(buffer=px_buf, bytes_per_row=W * 4), (W, H, 1))
device.queue.submit([enc.finish()])
# NOTE: queue.on_submitted_work_done_sync() is BROKEN in wgpu 0.32.0 (native callback
# ABI mismatch: 'void(*)(WGPUQueueWorkDoneStatus, WGPUStringView, void*, void*)' vs
# cffi 'void(*)(WGPUQueueWorkDoneStatus, void*, void*)' — TypeError in _api.py:4099).
# map_sync() below is itself a full fence for the preceding submits, so we rely on it.

# ---------- readback ----------
read_buf.map_sync(wgpu.MapMode.READ)
gpu_vals = struct.unpack("%df" % COUNT, read_buf.read_mapped().tobytes())
read_buf.unmap()
ref = ((np.arange(COUNT, dtype=np.uint32) ** 2) % 997).astype(np.float32)
diff = [i for i in range(COUNT) if np.float32(gpu_vals[i]) != ref[i]]
(ROOT / "buf_check.txt").write_text(
    "pattern=(i*i)%%997 count=%d mismatches=%d %s\n" % (COUNT, len(diff), diff[:10])
    + "first10_gpu=%s\nfirst10_ref=%s\n" % (gpu_vals[:10], ref[:10].tolist()))

px_buf.map_sync(wgpu.MapMode.READ)
px = np.frombuffer(px_buf.read_mapped().tobytes(), dtype=np.uint8).reshape(H, W, 4)
px_buf.unmap()
(ROOT / "sphere.ppm").write_bytes(b"P6\n256 256\n255\n" + px[:, :, :3].tobytes())

print("adapter info:", info)  # GPUAdapterInfo fields are adapter-specific
print("WGSL compiled by naga inside libwgpu_native (no external toolchain)")
print("WALL_SECONDS=%.2f" % (time.perf_counter() - t0))
print("OK")