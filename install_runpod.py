#!/usr/bin/env python3
"""
Installation script for ComfyUI-SAM3.
Updated to support comfy-env v0.1.x changes.
"""

import sys
import inspect
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ComfyUI-SAM3 Installation")
    print("=" * 60)

    # 1. Import install, handling potential missing IsolatedEnvManager
    try:
        from comfy_env import install
    except ImportError:
        # Fallback if the package structure is very old/broken
        try:
            from comfy_env import install
        except ImportError as e:
            print(f"\n[SAM3] Critical Error: Could not import 'install' from comfy_env. {e}")
            return 1

    node_root = Path(__file__).parent.absolute()
    config_path = node_root / "comfy-env.toml"

    # 2. Inspect the install() function signature
    # This detects what arguments the installed version of comfy-env actually accepts
    sig = inspect.signature(install)
    params = sig.parameters
    print(f"[SAM3] Detected comfy-env API version: {sig}")

    kwargs = {}

    # 3. Dynamically build arguments based on available parameters
    if 'config' in params:
        kwargs['config'] = config_path
    
    if 'mode' in params:
        kwargs['mode'] = "isolated"
        
    if 'node_dir' in params:
        kwargs['node_dir'] = node_root

    # 4. Execute installation
    try:
        # Scenario A: The function accepts standard kwargs (old API or compatible API)
        if kwargs:
            print(f"[SAM3] Installing with arguments: {list(kwargs.keys())}")
            install(**kwargs)
            
        # Scenario B: The function accepts arguments but we found no matching keywords
        # (New API likely just takes the path as a positional argument)
        else:
            print(f"[SAM3] No legacy arguments found. Attempting positional installation with: {node_root}")
            install(node_root)

    except TypeError as e:
        print(f"\n[SAM3] API Mismatch Warning: {e}")
        print("[SAM3] Attempting fallback to simple path installation...")
        try:
            # Fallback: Try passing just the path (works for some intermediate versions)
            install(str(node_root))
        except Exception as e2:
            print(f"\n[SAM3] Installation FAILED: {e2}")
            print("[SAM3] Report issues at: https://github.com/PozzettiAndrea/ComfyUI-SAM3/issues")
            return 1
            
    except Exception as e:
        print(f"\n[SAM3] Installation FAILED: {e}")
        return 1

    print("\n" + "=" * 60)
    print("[SAM3] Installation completed!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())