import sys
from huggingface_hub import model_info
from pynvml import (
    nvmlInit,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlShutdown,
)


def fmt(b):
    # format bytes to human-readable string
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024


def get_model_size(model_id, token=None):
    # get model weight size in bytes from HuggingFace (no download)
    info = model_info(model_id, files_metadata=True, token=token)
    total = 0
    for sib in info.siblings or []:
        if sib.rfilename.endswith((".safetensors", ".bin")):
            if sib.size is not None:
                total += sib.size
    if total == 0:
        sys.exit(f"error: could not determine size for '{model_id}'")
    return total


def get_gpus():
    # return list of (name, free_bytes, total_bytes) for each GPU
    nvmlInit()
    gpus = []
    for i in range(nvmlDeviceGetCount()):
        h = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(h)
        if isinstance(name, bytes):
            name = name.decode()
        mem = nvmlDeviceGetMemoryInfo(h)
        gpus.append((name, mem.free, mem.total))
    nvmlShutdown()
    return gpus


def main():
    args = sys.argv[1:]
    token = None

    if "--token" in args:
        idx = args.index("--token")
        if idx + 1 >= len(args):
            sys.exit("error: --token requires a value")
        token = args.pop(idx + 1)
        args.pop(idx)

    if not args or args[0] in ("-h", "--help"):
        print("usage: fitgpu <model_id> [--token TOKEN]")
        print("  e.g. fitgpu google/gemma-2-2b")
        sys.exit(0)

    model_id = args[0]
    print(f"model : {model_id}")

    size = get_model_size(model_id, token=token)
    print(f"size  : {fmt(size)} (weights on disk)")

    try:
        gpus = get_gpus()
    except Exception as e:
        sys.exit(f"error: could not query GPUs: {e}")

    if not gpus:
        sys.exit("error: no GPUs found")

    print()
    for i, (name, free, total) in enumerate(gpus):
        fits = "fits" if free >= size else "won't fit"
        print(f"GPU {i}: {name}")
        print(f"  VRAM : {fmt(total)} total, {fmt(free)} free")
        print(f"  result: {fits}")


if __name__ == "__main__":
    main()
