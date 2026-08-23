"""
Sinh 4 script driver cho thi nghiem CHOT "canonical vs drift-conditioning",
2 bien the x itr=3, chia deu tren 2 T4 trong 2 session rieng biet:

  Session 1 (Exchange + AirQuality):  ssf_final_s1_gpu0.sh / ssf_final_s1_gpu1.sh
  Session 2 (Jiaolong):               ssf_final_s2_gpu0.sh / ssf_final_s2_gpu1.sh

Vi sao chia theo CHI PHI (LPT) chu khong round-robin nhu make_kaggle_split.py?
Hom truoc moi job gan nhu dong deu (cung dataset Exchange, cung so buoc test), nen
round-robin la du. Lan nay chi phi lech manh theo 3 truc:
  * dataset : AirQuality co ~4115 buoc test/o vs Exchange ~3319 (va C=12 vs C=8)
  * seq_len : 1536 dat hon 512 khoang 1.8x (nhieu patch hon)
  * bien the: Proceed ~43.1 ms/buoc online vs canonical ~15.5 ms/buoc (do chay
              hypernetwork + concept encoder moi buoc) -> lech ~2.8x
Round-robin tren tap khong dong deu se de mot GPU rong som. Nen dung LPT
(longest-processing-time-first): uoc luong chi phi tung job, sap xep giam dan,
lan luot gan vao GPU dang nhe tai hon.

Mo hinh chi phi hieu chinh tu log smoke test THAT (MX450, Exchange 512/96):
  canonical: 64.5 it/s online (15.5 ms/buoc), Phase1 25 epoch x 6.3s
  proceed  : 23.2 it/s online (43.1 ms/buoc), Phase1 ~6 epoch (early-stop) x 19.4s
  so buoc test = int(n*0.45) - pred + 1   (khop CHINH XAC 3319 do duoc o 512/96)
Mo hinh nay chi dung de CHIA TAI (can bang tuong doi), khong phai de bao cao
thoi gian tuyet doi -- sai so tuyet doi +-40% khong anh huong den chat luong chia.

Moi cau hinh chay tron ven trong MOT invocation (--itr 3 chay 3 seed noi bo),
nen ca 3 seed cua cung mot cau hinh luon o CUNG mot GPU => std do duoc la
seed-variance thuan, khong lan cross-GPU divergence (xem thao luan ve viec
online phase 3319 buoc SGD tuan tu khuech dai sai khac floating-point).

Log dat ten rieng (ssffinal_..._itr3.log) de KHONG de len ket qua itr=1 cu
trong logs/ablation-study-2/ssfversion_*.log.

Usage:
    python3 ttm-scripts/ablation-study-2/make_ssf_final_split.py
Session 1 (Kaggle "GPU T4 x2"):
    bash ttm-scripts/ablation-study-2/ssf_final_s1_gpu0.sh &
    bash ttm-scripts/ablation-study-2/ssf_final_s1_gpu1.sh &
    wait
Session 2 (Kaggle "GPU T4 x2"):
    bash ttm-scripts/ablation-study-2/ssf_final_s2_gpu0.sh &
    bash ttm-scripts/ablation-study-2/ssf_final_s2_gpu1.sh &
    wait
"""
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = "logs/ablation-study-2"
ITR = 3

SEQ_LENS = [512, 1024, 1536]

# name -> (n_rows, n_channels, freq, pred_lens)
DATASETS = {
    "Exchange":      (7587,  8,  "d", [96, 192, 336]),
    "AirQuality":    (9357,  12, "h", [96, 192, 336]),
    "Jiaolong_DSMS": (30000, 24, "s", [96, 192, 336, 720]),
}

SESSIONS = {
    "s1": ["Exchange", "AirQuality"],
    "s2": ["Jiaolong_DSMS"],
}

# variant -> (log tag, run.py flags)
VARIANTS = {
    "beta_gamma": ("--online_method Proceed --concept_mode dual --concept_dim 32 "
                   "--bottleneck_dim 32 --ema 0 --ssf_no_input_scale"),
    "canonical":  "--online_method CanonicalSSF --tune_mode down_up",
}

COMMON = ("--model TinyTimeMixer --freeze_online --decoder_mode common_channel "
          "--train_ratio 0.5 --test_ratio 0.45 --batch_size 64 "
          f"--online_learning_rate 0.000003 --itr {ITR}")

# ---- he so mo hinh chi phi (xem docstring) ----
MS_ONLINE = {"canonical": 15.5, "beta_gamma": 43.1}   # ms / buoc online @ C=8, seq=512
EPOCHS_P1 = {"canonical": 25, "beta_gamma": 6}        # epoch Phase 1
SEC_PER_TRAIN_STEP = {"canonical": 6.3 / 49, "beta_gamma": 19.4 / 49}
F_CHANNEL = {7: 0.95, 8: 1.0, 12: 1.3, 24: 2.0}       # sub-linear: batch=1 -> nhieu overhead
F_SEQ = {512: 1.0, 1024: 1.4, 1536: 1.8}
# T4 nhanh hon MX450 (noi hieu chinh mo hinh) bao nhieu lan.
# Online phase chay batch=1 tuan tu -> bi chan boi kernel-launch/Python overhead chu
# khong phai FLOPS, nen loi it (2x). Phase 1 chay batch=64 -> compute-bound, loi nhieu (3.5x).
T4_SPEEDUP_ONLINE = 2.0
T4_SPEEDUP_TRAIN = 3.5


def job_cost(data, seq_len, pred_len, variant):
    n, C, _, _ = DATASETS[data]
    n_test, n_train = int(n * 0.45), int(n * 0.50)
    n_val = n - n_test - n_train
    test_steps = n_test - pred_len + 1
    val_steps = max(n_val - pred_len + 1, 1)
    train_steps = max(n_train // 64, 1)
    scale = F_CHANNEL[C] * F_SEQ[seq_len]
    online = ((test_steps + val_steps) * MS_ONLINE[variant] / 1000 * scale) / T4_SPEEDUP_ONLINE
    train = (EPOCHS_P1[variant] * train_steps * SEC_PER_TRAIN_STEP[variant] * scale) / T4_SPEEDUP_TRAIN
    return (online + train + 20) * ITR


def build_jobs(datasets):
    jobs = []
    for data in datasets:
        _, _, freq, preds = DATASETS[data]
        for variant, vflags in VARIANTS.items():
            for seq_len in SEQ_LENS:
                for pred_len in preds:
                    log = f"{LOG_DIR}/ssffinal_{data}_{seq_len}_{pred_len}_{variant}_itr{ITR}.log"
                    tag = f"[ssf-final] {data} seq={seq_len} pred={pred_len} variant={variant}"
                    flags = (f'--dataset "{data}" --seq_len {seq_len} --pred_len {pred_len} '
                             f'--freq {freq} {vflags}')
                    jobs.append(dict(log=log, tag=tag, flags=flags,
                                     cost=job_cost(data, seq_len, pred_len, variant)))
    return jobs


def lpt_split(jobs, n_bins=2):
    """Longest-processing-time-first: gan job nang nhat vao bin nhe nhat."""
    bins = [[] for _ in range(n_bins)]
    loads = [0.0] * n_bins
    for job in sorted(jobs, key=lambda j: -j["cost"]):
        i = loads.index(min(loads))
        bins[i].append(job)
        loads[i] += job["cost"]
    return bins, loads


def emit(session, gpu_id, jobs, load_sec):
    path = os.path.join(OUT_DIR, f"ssf_final_{session}_gpu{gpu_id}.sh")
    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by make_ssf_final_split.py -- session {session}, GPU {gpu_id}.",
        f"# {len(jobs)} cau hinh (moi cai --itr {ITR}); tai uoc luong ~{load_sec/3600:.1f} h.",
        "# Chia bang LPT theo chi phi uoc luong; do not hand-edit -- sua generator roi chay lai.",
        "set -u",
        f'if [ ! -d "./{LOG_DIR}" ]; then mkdir -p "./{LOG_DIR}"; fi',
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"GPU={gpu_id}",
        "",
    ]
    for j in jobs:
        log = j["log"]
        # skip chi khi da du ITR dong ket qua (itr=3 -> 3 dong 'mse:')
        lines += [
            f'if [ -s "{log}" ] && [ "$(grep -c \'mse:\' "{log}" 2>/dev/null)" -ge {ITR} ]; then',
            f'  echo "SKIP (done) {log}"',
            "else",
            f'  echo "{j["tag"]} (gpu=$GPU)"',
            f'  python run.py {COMMON} --gpu $GPU {j["flags"]} >> "{log}" 2>&1',
            "fi",
        ]
    lines.append(f'echo "[{session}/gpu{gpu_id}] DONE"')
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def main():
    for session, datasets in SESSIONS.items():
        jobs = build_jobs(datasets)
        bins, loads = lpt_split(jobs)
        total = sum(loads)
        print(f"\n=== session {session}: {', '.join(datasets)} ===")
        print(f"  {len(jobs)} cau hinh x itr={ITR} = {len(jobs)*ITR} lan chay")
        print(f"  tong uoc luong {total/3600:.1f} h (1 GPU) -> {max(loads)/3600:.1f} h (2 GPU song song)")
        for gpu_id, (jl, load) in enumerate(zip(bins, loads)):
            p = emit(session, gpu_id, jl, load)
            print(f"  gpu{gpu_id}: {len(jl):>2} cau hinh, ~{load/3600:4.1f} h -> {os.path.basename(p)}")
        imbalance = (max(loads) - min(loads)) / max(loads) * 100
        print(f"  lech tai giua 2 GPU: {imbalance:.1f}%")


if __name__ == "__main__":
    main()
