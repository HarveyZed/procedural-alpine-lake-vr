#!/bin/bash
# Why this wrapper exists: /usr/bin/time is a SIP-protected binary, so dyld strips
# DYLD_LIBRARY_PATH from its environment (and from /usr/bin/time's children's inherited
# env). Setting the vars INSIDE this script (exec'd as time's child, a fresh assignment
# in a protected shell that then execs UNPROTECTED python) restores discovery of
# tools/vk/lib/libvulkan.dylib. Without it: `import vulkan` OSError under /usr/bin/time.
cd "/path/to/procedural-alpine-lake" || exit 1
source tools/vk/env.sh
exec .venv/bin/python spike/p0_a/run.py "$@"
