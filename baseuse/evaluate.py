"""BaseUSE evaluation entry point.

Wraps the URGENT objective metric scripts in baseuse/evaluation/ and runs the
selected metrics sequentially, then prints/collects all RESULTS.txt summaries.

Usage:
    python -m baseuse.evaluate \
        --inf_scp ./exp/enhanced/inf.scp \
        --ref_scp /path/to/clean.scp \
        --output_dir ./exp/scores \
        --metrics intrusive,dnsmos,nisqa,utmos,speechbert,lps,spksim,wer \
        --meta_tsv /path/to/meta.tsv --utt2lang /path/to/utt2lang \
        --nj 8 --device cuda

Metric groups:
    intrusive   : PESQ / ESTOI / SDR / MCD / LSD          (CPU, needs ref_scp)
    dnsmos      : DNSMOS OVRL                              (needs DNSMOS onnx models)
    nisqa       : NISQA MOS                                (needs NISQA weights in evaluation/lib/NISQA)
    utmos       : UTMOS
    scoreq      : SCOREQ                                   (needs scoreq weights in evaluation/lib/scoreq)
    speechbert  : SpeechBERTScore                          (GPU, needs ref_scp)
    lps         : PhonemeSimilarity                        (GPU, needs ref_scp + eSpeak-NG)
    spksim      : SpeakerSimilarity                        (GPU, needs ref_scp)
    wer         : CER via OWSM                             (GPU, needs meta_tsv + utt2lang)

Each metric writes <output_dir>/<metric>/... with a RESULTS.txt when finished.
"""

import argparse
import os
import subprocess
import sys

EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
# run metric scripts with the repo root as cwd so that relative scp/output paths
# given on the CLI resolve against the user's current directory convention
ROOT_DIR = os.path.dirname(os.path.dirname(EVAL_DIR))


def build_commands(args, metric):
    out = os.path.join(args.output_dir, metric)
    py = sys.executable
    script = os.path.join(EVAL_DIR, SCRIPTS[metric])

    if metric == "intrusive":  # CPU only, no --device
        return [py, script, "--ref_scp", args.ref_scp, "--inf_scp", args.inf_scp,
                "--output_dir", out, "--nj", str(args.nj)]

    if metric in ("dnsmos", "nisqa", "utmos", "scoreq"):
        cmd = [py, script, "--inf_scp", args.inf_scp, "--output_dir", out,
               "--device", args.device]
        if metric == "nisqa" and args.nisqa_model:
            cmd += ["--nisqa_model", args.nisqa_model]
        if metric == "dnsmos" and args.dnsmos_dir != "none":
            cmd += ["--primary_model", os.path.join(args.dnsmos_dir, "sig_bak_ovr.onnx"),
                    "--p808_model", os.path.join(args.dnsmos_dir, "model_v8.onnx")]
        return cmd

    if metric in ("speechbert", "lps", "spksim"):
        return [py, script, "--ref_scp", args.ref_scp, "--inf_scp", args.inf_scp,
                "--output_dir", out, "--device", args.device]

    if metric == "wer":
        return [py, script, "--meta_tsv", args.meta_tsv, "--utt2lang", args.utt2lang,
                "--inf_scp", args.inf_scp, "--output_dir", out, "--device", args.device]

    raise KeyError(metric)


SCRIPTS = {
    "intrusive": "calculate_intrusive_se_metrics.py",
    "dnsmos": "calculate_nonintrusive_dnsmos.py",
    "nisqa": "calculate_nonintrusive_nisqa.py",
    "utmos": "calculate_nonintrusive_utmos.py",
    "scoreq": "calculate_nonintrusive_scoreq.py",
    "speechbert": "calculate_speechbert_score.py",
    "lps": "calculate_phoneme_similarity.py",
    "spksim": "calculate_speaker_similarity.py",
    "wer": "calculate_wer.py",
}

NEEDS_REF = {"intrusive", "speechbert", "lps", "spksim"}
NEEDS_META = {"wer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--inf_scp", type=str, required=True,
                        help="2-column scp of enhanced audio (uid wav_path)")
    parser.add_argument("--ref_scp", type=str, default="none",
                        help="2-column scp of clean reference audio")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metrics", type=str,
                        default="intrusive,dnsmos,nisqa,utmos",
                        help=f"Comma-separated subset of {sorted(SCRIPTS)}")
    parser.add_argument("--meta_tsv", type=str, default="none",
                        help="meta.tsv (id ... text) for CER")
    parser.add_argument("--utt2lang", type=str, default="none",
                        help="utt2lang file for CER")
    parser.add_argument("--nj", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nisqa_model", type=str,
                        default=os.path.join(EVAL_DIR, "lib", "NISQA", "weights", "nisqa.tar"))
    parser.add_argument("--dnsmos_dir", type=str, default="none",
                        help="Directory containing sig_bak_ovr.onnx and model_v8.onnx")
    args = parser.parse_args()

    # normalize all path args to absolute so the metric subprocesses (cwd=ROOT_DIR)
    # resolve them identically regardless of where evaluate.py was launched from
    for p in ("inf_scp", "ref_scp", "output_dir", "meta_tsv", "utt2lang"):
        v = getattr(args, p)
        if v and v != "none":
            setattr(args, p, os.path.abspath(v))

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in SCRIPTS]
    assert not unknown, f"unknown metrics: {unknown} (available: {sorted(SCRIPTS)})"

    for m in metrics:
        if m in NEEDS_REF:
            assert args.ref_scp != "none", f"--ref_scp is required for metric: {m}"
        if m in NEEDS_META:
            assert args.meta_tsv != "none" and args.utt2lang != "none", \
                f"--meta_tsv and --utt2lang are required for metric: {m}"

    os.makedirs(args.output_dir, exist_ok=True)

    for metric in metrics:
        cmd = build_commands(args, metric)
        print(f"\n=== [{metric}] {' '.join(cmd)}\n", flush=True)
        ret = subprocess.run(cmd, cwd=ROOT_DIR).returncode
        if ret != 0:
            print(f"[{metric}] FAILED (exit {ret})", flush=True)
        else:
            print(f"[{metric}] done", flush=True)

    # collect RESULTS.txt
    print("\n================ SUMMARY ================")
    for metric in metrics:
        result_dir = os.path.join(args.output_dir, metric)
        found = False
        for root, _, files in os.walk(result_dir):
            for fn in files:
                if fn == "RESULTS.txt":
                    path = os.path.join(root, fn)
                    print(f"--- {path}")
                    with open(path) as f:
                        print(f.read().strip())
                    found = True
        if not found:
            print(f"--- [{metric}] no RESULTS.txt found")


if __name__ == "__main__":
    main()
