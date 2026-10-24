#!/usr/bin/env python3
"""P0 Spike A — MoltenVK via pure-python vulkan bindings (1.3.275.1).

Scene + fill pattern are LOCKED and shared with spike/p0_b and spike/p0_ref.py
(see shaders/*.glsl header comments). 256x256 single frame, compute + one
fullscreen-triangle fragment raymarch of a lit sphere. Run after:
    source tools/vk/env.sh
Outputs: sphere.ppm (P6), buf_check.txt, WALL_SECONDS line.
"""
import subprocess, sys, time, pathlib, struct
import vulkan as vk

ROOT = pathlib.Path(__file__).resolve().parent          # spike/p0_a
WS = ROOT.parents[1]                                     # workspace root
GLSLC = WS / "tools/vk/bin/glslc"
W = H = 256
COUNT = 1024
VK_API_1_2 = (1 << 22) | (2 << 12)
KEEP = []   # keep cdata alive (shaders etc.)

t0 = time.perf_counter()


def compile_glsl(src, out):
    subprocess.run([str(GLSLC), "--target-env=vulkan1.2", str(src), "-o", str(out)],
                   check=True, cwd=str(WS))
    data = out.read_bytes()
    mod = vk.vkCreateShaderModule(device, vk.VkShaderModuleCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        codeSize=len(data),
        pCode=data), None)  # ctor casts bytes -> uint32_t* itself
    KEEP.append(data)
    return mod


def find_memtype(props, need):
    for i in range(props.memoryTypeCount):
        if (props.memoryTypes[i].propertyFlags & need) == need:
            return i
    raise RuntimeError("no memtype for " + hex(need))


# ---- instance (portability enumeration: MoltenVK ICD is flagged portability) ----
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    flags=vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
    pApplicationInfo=vk.VkApplicationInfo(apiVersion=VK_API_1_2, pApplicationName="p0a",
        applicationVersion=1, pEngineName="p0a", engineVersion=1),
    ppEnabledExtensionNames=[b"VK_KHR_portability_enumeration"]), None)

pd = vk.vkEnumeratePhysicalDevices(inst)[0]
dev_name = vk.vkGetPhysicalDeviceProperties(pd).deviceName
memprops = vk.vkGetPhysicalDeviceMemoryProperties(pd)
qfs = vk.vkGetPhysicalDeviceQueueFamilyProperties(pd)
qfi = next(i for i, q in enumerate(qfs)
           if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT and q.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT)

dev_exts = [e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(pd, None)]  # auto str
enable = [b"VK_KHR_portability_subset"] if "VK_KHR_portability_subset" in dev_exts else []

device = vk.vkCreateDevice(pd, vk.VkDeviceCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    queueCreateInfoCount=1,
    pQueueCreateInfos=vk.VkDeviceQueueCreateInfo(
        queueFamilyIndex=qfi, queueCount=1, pQueuePriorities=[1.0]),
    ppEnabledExtensionNames=enable), None)
queue = vk.vkGetDeviceQueue(device, qfi, 0)

# ---- buffers ----
def make_buffer(size, usage, need):
    buf = vk.vkCreateBuffer(device, vk.VkBufferCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, size=size,
        usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
    req = vk.vkGetBufferMemoryRequirements(device, buf)
    mt = find_memtype(memprops, need)
    mem = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(
        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        allocationSize=req.size, memoryTypeIndex=mt), None)
    vk.vkBindBufferMemory(device, buf, mem, 0)
    return buf, mem

DEV_LOCAL = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
HOST_COH = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT

comp_buf, comp_mem = make_buffer(COUNT * 4, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                 | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, DEV_LOCAL)
stg_buf, stg_mem = make_buffer(COUNT * 4, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_COH)
px_buf, px_mem = make_buffer(W * H * 4, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_COH)

# ---- image ----
img = vk.vkCreateImage(device, vk.VkImageCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    extent=vk.VkExtent3D(width=W, height=H, depth=1), mipLevels=1, arrayLayers=1,
    samples=vk.VK_SAMPLE_COUNT_1_BIT, tiling=vk.VK_IMAGE_TILING_OPTIMAL,
    usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
req = vk.vkGetImageMemoryRequirements(device, img)
img_mem = vk.vkAllocateMemory(device, vk.VkMemoryAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    allocationSize=req.size, memoryTypeIndex=find_memtype(memprops, DEV_LOCAL)), None)
vk.vkBindImageMemory(device, img, img_mem, 0)
view = vk.vkCreateImageView(device, vk.VkImageViewCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, image=img,
    viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM,
    subresourceRange=vk.VkImageSubresourceRange(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1)), None)

# ---- compute pipeline ----
cs = compile_glsl(ROOT / "shaders/fill.comp", ROOT / "fill.spv")
dsl = vk.vkCreateDescriptorSetLayout(device, vk.VkDescriptorSetLayoutCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pBindings=[
        vk.VkDescriptorSetLayoutBinding(binding=0, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                        descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT)]), None)
pl_cmp = vk.vkCreatePipelineLayout(device, vk.VkPipelineLayoutCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, pSetLayouts=[dsl]), None)
pipe_cmp = vk.vkCreateComputePipelines(device, vk.VK_NULL_HANDLE, 1, vk.VkComputePipelineCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    stage=vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=cs, pName=b"main"),
    layout=pl_cmp), None)[0]

pool = vk.vkCreateDescriptorPool(device, vk.VkDescriptorPoolCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, maxSets=1, pPoolSizes=[
        vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1)]), None)
ds = vk.vkAllocateDescriptorSets(device, vk.VkDescriptorSetAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    descriptorPool=pool, pSetLayouts=[dsl]))[0]
bi = vk.VkDescriptorBufferInfo(buffer=comp_buf, offset=0, range=COUNT * 4)
vk.vkUpdateDescriptorSets(device, 1, [vk.VkWriteDescriptorSet(
    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, dstSet=ds, dstBinding=0,
    descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=bi)], 0, None)

# ---- render pass + graphics pipeline ----
rp = vk.vkCreateRenderPass(device, vk.VkRenderPassCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    pAttachments=[vk.VkAttachmentDescription(
        format=vk.VK_FORMAT_R8G8B8A8_UNORM, samples=vk.VK_SAMPLE_COUNT_1_BIT,
        loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
        stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE, stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)],
    pSubpasses=[vk.VkSubpassDescription(
        pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
        pColorAttachments=[vk.VkAttachmentReference(attachment=0,
                          layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)])],
    dependencyCount=0), None)
fb = vk.vkCreateFramebuffer(device, vk.VkFramebufferCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, renderPass=rp,
    pAttachments=[view], width=W, height=H, layers=1), None)

vs = compile_glsl(ROOT / "shaders/fs.vert", ROOT / "fs.vert.spv")
fs = compile_glsl(ROOT / "shaders/sphere.frag", ROOT / "sphere.spv")
pl_gfx = vk.vkCreatePipelineLayout(device, vk.VkPipelineLayoutCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO), None)
pipe_gfx = vk.vkCreateGraphicsPipelines(device, vk.VK_NULL_HANDLE, 1, vk.VkGraphicsPipelineCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    pStages=[vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vs, pName=b"main"),
        vk.VkPipelineShaderStageCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=fs, pName=b"main")],
    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO),
    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
    pViewportState=vk.VkPipelineViewportStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewportCount=1, scissorCount=1),
    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=vk.VK_CULL_MODE_NONE,
        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE, lineWidth=1.0),
    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT),
    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        pAttachments=[vk.VkPipelineColorBlendAttachmentState(
            colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT
            | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT)]),
    pDynamicState=vk.VkPipelineDynamicStateCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        pDynamicStates=[vk.VK_DYNAMIC_STATE_VIEWPORT, vk.VK_DYNAMIC_STATE_SCISSOR]),
    layout=pl_gfx, renderPass=rp, subpass=0), None)[0]

# ---- command buffer ----
poolc = vk.vkCreateCommandPool(device, vk.VkCommandPoolCreateInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, queueFamilyIndex=qfi), None)
cmd = vk.vkAllocateCommandBuffers(device, vk.VkCommandBufferAllocateInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, commandPool=poolc,
    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(
    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT))

# compute: fill COUNT floats
vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe_cmp)
vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl_cmp, 0, 1, [ds], 0, [])
vk.vkCmdDispatch(cmd, COUNT // 16, 1, 1)
# compute write -> transfer read
vk.vkCmdPipelineBarrier(cmd, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, [], 1, [
        vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=comp_buf, size=COUNT * 4)], 0, [])
vk.vkCmdCopyBuffer(cmd, comp_buf, stg_buf, 1, [vk.VkBufferCopy(size=COUNT * 4)])

# graphics: raymarch sphere into image (render pass does UNDEFINED->COLOR, final->TRANSFER_SRC)
clr = vk.ffi.new("VkClearValue[1]")
clr[0].color.float32 = [0.0, 0.0, 0.0, 1.0]
vk.vkCmdBeginRenderPass(cmd, vk.VkRenderPassBeginInfo(
    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, renderPass=rp, framebuffer=fb,
    renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=W, height=H)),
    pClearValues=clr), vk.VK_SUBPASS_CONTENTS_INLINE)
vk.vkCmdSetViewport(cmd, 0, 1, [vk.VkViewport(x=0.0, y=0.0, width=float(W), height=float(H),
                                              minDepth=0.0, maxDepth=1.0)])
vk.vkCmdSetScissor(cmd, 0, 1, [vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0),
                                           extent=vk.VkExtent2D(width=W, height=H))])
vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipe_gfx)
vk.vkCmdDraw(cmd, 3, 1, 0, 0)
vk.vkCmdEndRenderPass(cmd)
vk.vkCmdCopyImageToBuffer(cmd, img, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, px_buf, 1, [
    vk.VkBufferImageCopy(
        imageSubresource=vk.VkImageSubresourceLayers(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1, mipLevel=0),
        imageExtent=vk.VkExtent3D(width=W, height=H, depth=1))])
vk.vkEndCommandBuffer(cmd)

vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(
    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, pCommandBuffers=[cmd])], None)
vk.vkQueueWaitIdle(queue)

# ---- readback ----
buf = vk.vkMapMemory(device, stg_mem, 0, COUNT * 4, 0)
gpu_vals = list(struct.unpack("%df" % COUNT, bytes(buf)))
vk.vkUnmapMemory(device, stg_mem)
import numpy as np
ref = ((np.arange(COUNT, dtype=np.uint32) ** 2) % 997).astype(np.float32)
diff = [i for i in range(COUNT) if np.float32(gpu_vals[i]) != ref[i]]
(ROOT / "buf_check.txt").write_text(
    "pattern=(i*i)%%997 count=%d mismatches=%d %s\n" % (COUNT, len(diff), diff[:10])
    + "first10_gpu=%s\nfirst10_ref=%s\n" % (gpu_vals[:10], ref[:10].tolist()))

pbuf = vk.vkMapMemory(device, px_mem, 0, W * H * 4, 0)
px = bytes(pbuf)
vk.vkUnmapMemory(device, px_mem)
rgb = bytearray()
for i in range(W * H):
    rgb += px[i * 4:i * 4 + 3]
(ROOT / "sphere.ppm").write_bytes(b"P6\n256 256\n255\n" + bytes(rgb))

# teardown
vk.vkDeviceWaitIdle(device)
vk.vkDestroyImageView(device, view, None)
vk.vkDestroyFramebuffer(device, fb, None)
vk.vkDestroyRenderPass(device, rp, None)
vk.vkDestroyImage(device, img, None)
vk.vkFreeMemory(device, img_mem, None)
vk.vkDestroyPipeline(device, pipe_gfx, None)
vk.vkDestroyPipeline(device, pipe_cmp, None)
vk.vkDestroyPipelineLayout(device, pl_gfx, None)
vk.vkDestroyPipelineLayout(device, pl_cmp, None)
vk.vkDestroyDescriptorPool(device, pool, None)
vk.vkDestroyShaderModule(device, vs, None)
vk.vkDestroyShaderModule(device, fs, None)
vk.vkDestroyShaderModule(device, cs, None)
vk.vkDestroyBuffer(device, comp_buf, None); vk.vkFreeMemory(device, comp_mem, None)
vk.vkDestroyBuffer(device, stg_buf, None); vk.vkFreeMemory(device, stg_mem, None)
vk.vkDestroyBuffer(device, px_buf, None);  vk.vkFreeMemory(device, px_mem, None)
vk.vkDestroyCommandPool(device, poolc, None)
vk.vkDestroyDevice(device, None)
vk.vkDestroyInstance(inst, None)

print("device:", dev_name)
print("WALL_SECONDS=%.2f" % (time.perf_counter() - t0))
print("OK")