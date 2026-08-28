import statistics
import torch

from yolox.exp import get_exp

EXP = "exps/custom/yolox_m_traffic_nature_13.py"
CKPT = "YOLOX_outputs/yolox_m_traffic_nature_13_full/best_ckpt_model_only.pth"

WARMUP = 50
ITERS = 300

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

device = torch.device("cuda:0")

exp = get_exp(EXP, None)
model = exp.get_model()

ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model"], strict=True)

model = model.to(device).eval()

x = torch.rand(1, 3, 640, 640, device=device)


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

    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()

    median = statistics.median(times)
    mean = statistics.mean(times)
    p95 = times[int(0.95 * (len(times) - 1))]

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

    mode = "FP16" if fp16 else "FP32"

    print(f"\n{mode}")
    print(f"median_ms      : {median:.3f}")
    print(f"mean_ms        : {mean:.3f}")
    print(f"p95_ms         : {p95:.3f}")
    print(f"FPS_from_median: {1000.0 / median:.2f}")
    print(f"peak_alloc_MB  : {peak_alloc:.1f}")
    print(f"peak_reserved_MB: {peak_reserved:.1f}")


print("MODEL: YOLOX-M")
print("batch: 1")
print("input: 640x640")
print("warmup:", WARMUP)
print("iterations:", ITERS)

benchmark(False)
benchmark(True)
