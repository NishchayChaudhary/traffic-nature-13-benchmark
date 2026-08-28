import statistics
import torch

from src.core import YAMLConfig

CONFIG = "configs/custom/traffic_nature_13_rtdetrv2_s.yml"
CKPT = "output/traffic_nature_13_rtdetrv2_s_native640/best.pth"

WARMUP = 50
ITERS = 300

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda:0")

# Build model
cfg = YAMLConfig(CONFIG)

# Trusted checkpoint produced by our own training run
checkpoint = torch.load(
    CKPT,
    map_location="cpu",
    weights_only=False,
)

if "ema" in checkpoint and checkpoint["ema"] is not None:
    state = checkpoint["ema"]["module"]
    print("weights: EMA")
else:
    state = checkpoint["model"]
    print("weights: MODEL")

cfg.model.load_state_dict(state)

# Official RT-DETR deploy mode
model = cfg.model.deploy()
model = model.to(device).eval()

x = torch.rand(
    1, 3, 640, 640,
    device=device,
)


def benchmark(fp16):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=fp16,
        ):
            # Warmup
            for _ in range(WARMUP):
                _ = model(x)

            torch.cuda.synchronize()

            starts = []
            ends = []

            for _ in range(ITERS):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)

                s.record()
                _ = model(x)
                e.record()

                starts.append(s)
                ends.append(e)

            torch.cuda.synchronize()

    times = [
        s.elapsed_time(e)
        for s, e in zip(starts, ends)
    ]
    times.sort()

    median = statistics.median(times)
    mean = statistics.mean(times)
    p95 = times[int(0.95 * (len(times) - 1))]

    peak_alloc = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**2
    )

    mode = "FP16" if fp16 else "FP32"

    print(f"\n{mode}")
    print(f"median_ms      : {median:.3f}")
    print(f"mean_ms        : {mean:.3f}")
    print(f"p95_ms         : {p95:.3f}")
    print(f"FPS_from_median: {1000.0 / median:.2f}")
    print(f"peak_alloc_MB  : {peak_alloc:.1f}")
    print(f"peak_reserved_MB: {peak_reserved:.1f}")


print("MODEL: RT-DETRv2-S")
print("batch: 1")
print("input: 640x640")
print("warmup:", WARMUP)
print("iterations:", ITERS)

benchmark(False)
benchmark(True)
