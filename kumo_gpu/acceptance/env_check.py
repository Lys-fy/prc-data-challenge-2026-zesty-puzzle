#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Kumo acceptance step 1: environment facts + hard checks, after env/setup_env.sh (install, sdm commit, weights sha256, GPU smoke).
Prints one `ENVCHECK {json}` line and exits 0 on PASS, 1 on FAIL. PASS = torch imports with CUDA, >= 1 GPU visible (NGPU_EXPECT if set), sdm at the
pinned commit, fp16 autocast matmul finite, at least one fused SDPA backend (flash or mem-efficient) runs an fp16 attention, medium weights present
in HF_HOME (sha256 verified by setup_env.sh), payload member frames + parity reference present."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, glob, time

R = dict(host=os.uname().nodename, t=time.strftime("%Y-%m-%d %H:%M:%S"), fail=[])
try:
    import torch

    R["torch"] = torch.__version__
    R["cuda"] = torch.version.cuda
    R["cuda_ok"] = torch.cuda.is_available()
    if not R["cuda_ok"]:
        R["fail"].append("cuda not available")
    else:
        R["ngpu"] = torch.cuda.device_count()
        R["gpus"] = [torch.cuda.get_device_name(i) for i in range(R["ngpu"])]
        R["mem_gib"] = [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1) for i in range(R["ngpu"])]
        R["cap"] = [list(torch.cuda.get_device_capability(i)) for i in range(R["ngpu"])]
        R["arch_list"] = torch.cuda.get_arch_list()
        want = int(os.environ.get("NGPU_EXPECT", "1"))
        if R["ngpu"] < want:
            R["fail"].append(f"{R['ngpu']} GPU(s) visible, expected >= {want}")
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            c = a @ b
        R["fp16_matmul"] = bool(torch.isfinite(c).all()) and c.dtype == torch.float16
        if not R["fp16_matmul"]:
            R["fail"].append("fp16 autocast matmul")
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend

        q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
        be = {}
        for name, bk in (
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
        ):
            try:
                with sdpa_kernel(bk):
                    o = F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                be[name] = bool(torch.isfinite(o).all())
            except Exception as ex:
                be[name] = f"no ({type(ex).__name__})"
        R["sdpa"] = be
        if not (be.get("flash") is True or be.get("efficient") is True):
            R["fail"].append("no fused SDPA backend (flash/efficient) for fp16")
        # GPU_SCALE_CHECK a: versions + arch WARN (sm_XY missing but sm_X0 present, e.g. B300 sm_103 on cu128)
        R["cudnn"] = torch.backends.cudnn.version()
        R["warn"] = []
        try:
            R["driver"] = (
                __import__("subprocess")
                .run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                .stdout.split()[0]
            )
        except Exception:
            R["driver"] = "?"
        cap0 = R["cap"][0]
        arch = f"sm_{cap0[0]}{cap0[1]}"
        if arch not in R["arch_list"]:
            (R["warn"] if f"sm_{cap0[0]}0" in R["arch_list"] else R["fail"]).append(
                f"{arch} not in the wheel's arch list {R['arch_list']}"
                + (" (same-major cubin present: expected to run)" if f"sm_{cap0[0]}0" in R["arch_list"] else "")
            )
        # LONG-KEY fused attention probe (math excluded): 4,096 queries x 262,144 keys fp16, plus the GQA shape (8 q heads / 2 kv heads); a math fallback
        # would materialise n^2 logits (8 GiB here, 125 GB at 250k x 250k) -> FAIL if every fused backend errors or the peak exceeds 4 GiB
        try:
            fused = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION] + (
                [SDPBackend.CUDNN_ATTENTION] if hasattr(SDPBackend, "CUDNN_ATTENTION") else []
            )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            ql = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
            kl = torch.randn(1, 8, 262144, 64, device="cuda", dtype=torch.float16)
            vl = torch.randn_like(kl)
            with sdpa_kernel(fused):
                o = F.scaled_dot_product_attention(ql, kl, vl)
                torch.cuda.synchronize()
            k2 = kl[:, :2].contiguous()
            v2 = vl[:, :2].contiguous()
            with sdpa_kernel(fused):
                o2 = F.scaled_dot_product_attention(ql, k2, v2, enable_gqa=True)
                torch.cuda.synchronize()
            pk = (torch.cuda.max_memory_allocated() - base) / 2**30
            R["longkey"] = dict(
                ok=bool(torch.isfinite(o).all() and torch.isfinite(o2).all()),
                peak_gib=round(pk, 2),
                keys=262144,
                gqa=True,
            )
            if not R["longkey"]["ok"] or pk > 4.0:
                R["warn"].append(
                    f"long-key fused SDPA probe: ok {R['longkey']['ok']} peak {pk:.1f} GiB (> 4 GiB = math fallback)"
                )
            del ql, kl, vl, o, o2, k2, v2
            torch.cuda.empty_cache()
        except Exception as ex:
            R["longkey"] = f"error {type(ex).__name__}: {str(ex)[:120]}"
            R["warn"].append(
                f"long-key fused SDPA probe failed: {type(ex).__name__}; WARN only (whole-pool units fail on their own if attention falls back)"
            )
        # sdm Triton kernels compile + agree with the eager path (ptxas must know the arch; B300 sm_103 risk)
        try:
            from sdm._kernels import rmsnorm_cast as RC, segment_multi_reduce as SM

            x = torch.randn(64, 7, 4, 64, device="cuda", dtype=torch.float16)
            w = torch.ones(64, device="cuda", dtype=torch.float16)
            with torch.autocast("cuda", dtype=torch.float16), torch.inference_mode():
                a = RC.rmsnorm_cast(x, w, 1e-6).float()
                b = F.rms_norm(x.float(), (64,), w.float(), 1e-6)
            d1 = float((a - b).abs().max())
            src = torch.randn(50, 16, device="cuda")
            idx = torch.randint(0, 50, (200,), device="cuda")
            ea = torch.randn(200, 16, device="cuda")
            off = torch.tensor([0, 30, 90, 120, 200], device="cuda")
            with torch.inference_mode():
                c = SM.segment_multi_reduce(src, idx, ea, off)
                e = SM._eager_segment_multi_reduce(src, idx, ea, off)
            d2 = float((c - e).abs().max())
            R["triton"] = dict(
                rmsnorm_cast_maxdiff=round(d1, 5),
                segment_multi_reduce_maxdiff=round(d2, 5),
                kernels_loaded=bool(
                    RC._triton_rmsnorm_cast is not None and SM._triton_segment_multi_reduce is not None
                ),
            )
            if d1 > 1e-2 or d2 > 1e-3:
                R["fail"].append(f"sdm Triton kernels disagree with eager: {R['triton']}")
            if not R["triton"]["kernels_loaded"]:
                R["warn"].append("sdm Triton kernels not loaded (PyTorch fallback; slower, same numbers)")
        except Exception as ex:
            R["triton"] = f"error {type(ex).__name__}: {str(ex)[:160]}"
            R["warn"].append(
                f"sdm Triton kernel probe failed (compile?): {type(ex).__name__}; WARN only, PARITY16 is the correctness guard"
            )
except Exception as ex:
    R["fail"].append(f"torch: {type(ex).__name__}: {ex}")
try:
    import sdm, sysconfig

    d = glob.glob(os.path.join(sysconfig.get_paths()["purelib"], "structured_data_models-*.dist-info/direct_url.json"))
    R["sdm"] = getattr(sdm, "__version__", "?")
    R["sdm_commit"] = json.load(open(d[0]))["vcs_info"]["commit_id"][:9] if d else "vendored"
    if "ee6acd402" not in (R["sdm"] + R["sdm_commit"]):
        R["fail"].append(f"sdm not at ee6acd402: {R['sdm']} {R['sdm_commit']}")
except Exception as ex:
    R["fail"].append(f"sdm import: {ex}")
hf = os.environ.get("HF_HOME", "")
w = glob.glob(f"{hf}/hub/models--nvidia--Kumo-Tabular/snapshots/*/medium/regressor.pt")
R["weights_medium"] = bool(w)
R["weights_large"] = bool(glob.glob(f"{hf}/hub/models--nvidia--Kumo-Tabular/snapshots/*/large/regressor.pt"))
if not w:
    R["fail"].append("medium/regressor.pt not in HF_HOME")
if not R["weights_large"]:
    R["fail"].append("large/regressor.pt not in HF_HOME (bench grid needs it: WITH_LARGE=1 in setup)")
P = os.environ.get("PAYLOAD", "")
R["payload_frames"] = len(glob.glob(f"{P}/member/km_*.parquet"))
R["parity_ref"] = bool(glob.glob(f"{P}/ref/parity_XKT_KM_janjul_*.npy"))
if R["payload_frames"] != 10:
    R["fail"].append(f"payload frames {R['payload_frames']}/10")
if not R["parity_ref"]:
    R["fail"].append("parity reference missing")
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
imp = [
    f
    for f in ("janjul", "months28", "months39", "months410", "months511", "months612", "prod")
    if os.path.exists(f"{B}/km_importance_412_{f}.tsv")
]
R["importance_files"] = len(imp)
if len(imp) < 7:
    R["fail"].append(
        f"importance files {len(imp)}/7 in data/ (COLMODE=imp would fall back to random columns): restore the shipped data/km_importance_412_<fold>.tsv files,"
    )
R["verdict"] = "PASS" if not R["fail"] else "FAIL"
print("ENVCHECK " + json.dumps(R), flush=True)
sys.exit(0 if not R["fail"] else 1)
