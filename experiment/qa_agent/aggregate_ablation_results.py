"""消融实验结果汇总：读取各组运行输出，生成 CSV 和 Markdown 对比表。"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter


RUNS_DIR = Path("runs/qa_ablation")
SUMMARY_DIR = RUNS_DIR / "summary"

# ── 各组按顺序排列 ──
GROUP_ORDER = ["A_direct", "B_no_feedback", "C_feedback_no_diff", "D_full"]
GROUP_LABELS = {
    "A_direct": "A - Direct",
    "B_no_feedback": "B - No Feedback",
    "C_feedback_no_diff": "C - No Difficulty",
    "D_full": "D - Full",
}


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_metrics(run_dir: Path) -> dict:
    """从各输出文件中提取统计指标。"""
    metadata = load_json(run_dir / "metadata.json") or {}
    gen_report = load_json(run_dir / "generation_report.json") or {}
    candidates = load_json(run_dir / "candidate_pool.jsonl") or load_json(run_dir / "direct_questions.json") or []
    round_records = load_jsonl(run_dir / "round_summary.jsonl")
    llm_calls = load_jsonl(run_dir / "llm_calls.jsonl")

    # ── 基本指标 ──
    accepted_count = gen_report.get("candidate_count", 0) if gen_report else len(candidates)
    generated_count = gen_report.get("total_generated", 0) if gen_report else 0

    # 从 round_summary 汇总
    total_generated = 0
    total_accepted = 0
    strategies = []
    for r in round_records:
        total_generated += r.get("generated", 0) if "generated" in r else r.get("generated_count", 0)
        total_accepted += r.get("accepted", 0) if "accepted" in r else r.get("accepted_count", 0)
        strategies.append(r.get("strategy", "unknown"))

    if total_generated > 0:
        generated_count = total_generated
    if total_accepted > 0:
        accepted_count = total_accepted

    if generated_count > 0:
        accept_rate = accepted_count / max(1, generated_count)
    else:
        accept_rate = 0.0

    # ── 难度分布 ──
    diff_counts = gen_report.get("difficulty_counts", {}) if gen_report else {}
    total_diff = sum(diff_counts.values()) or accepted_count
    easy_ratio = diff_counts.get("easy", 0) / max(1, total_diff)
    medium_ratio = diff_counts.get("medium", 0) / max(1, total_diff)
    hard_ratio = diff_counts.get("hard", 0) / max(1, total_diff)

    # ── Token 统计 ──
    token_stats = gen_report.get("token_stats", {}) if gen_report else {}
    total_prompt_tokens = token_stats.get("input_tokens", 0)
    total_completion_tokens = token_stats.get("output_tokens", 0)
    total_tokens = total_prompt_tokens + total_completion_tokens
    total_llm_calls = token_stats.get("calls", 0)

    # 从 llm_calls.jsonl 汇总（如果有的话）
    if llm_calls:
        total_prompt_tokens = sum(c.get("input_tokens", 0) for c in llm_calls)
        total_completion_tokens = sum(c.get("output_tokens", 0) for c in llm_calls)
        total_tokens = total_prompt_tokens + total_completion_tokens
        total_llm_calls = len(llm_calls)

    tokens_per_accepted = total_tokens / max(1, accepted_count)

    # ── 策略分布 ──
    strategy_counts = Counter(strategies) if strategies else {}

    # ── multi-chunk 比例 ──
    multi_count = 0
    single_count = 0
    for c in (candidates if isinstance(candidates, list) else []):
        if isinstance(c, dict):
            if c.get("is_multi_chunk") or c.get("chunk_ids") or c.get("multi_chunk"):
                multi_count += 1
            else:
                single_count += 1
    multi_chunk_ratio = multi_count / max(1, multi_count + single_count)

    # ── 重复率（从 metadata 或 report） ──
    duplicate_ratio = gen_report.get("duplicate_ratio", 0.0) if gen_report else 0.0

    run_id = metadata.get("run_id", run_dir.name)
    group_id = metadata.get("group_id", run_id.split("_")[0])

    return {
        "run_id": run_id,
        "group_id": group_id,
        "seed": metadata.get("seed", "-"),
        "model_name": metadata.get("model_name", "-"),
        "count": metadata.get("count", 0),
        "max_rounds": metadata.get("max_rounds", 0),
        "topics": len(metadata.get("topics", [])),
        "accepted_count": accepted_count,
        "generated_count": generated_count,
        "accept_rate": round(accept_rate, 4),
        "easy_ratio": round(easy_ratio, 4),
        "medium_ratio": round(medium_ratio, 4),
        "hard_ratio": round(hard_ratio, 4),
        "multi_chunk_ratio": round(multi_chunk_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "total_llm_calls": total_llm_calls,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_accepted": round(tokens_per_accepted, 1),
        "strategy_counts": dict(strategy_counts),
    }


def compute_deltas(metrics: dict[str, dict]) -> dict[str, dict]:
    """计算消融对比差值。"""
    deltas = {}
    pairs = [
        ("B - A", "B_no_feedback", "A_direct"),
        ("C - B", "C_feedback_no_diff", "B_no_feedback"),
        ("D - C", "D_full", "C_feedback_no_diff"),
    ]
    numerical_keys = [
        "accepted_count", "generated_count", "accept_rate",
        "easy_ratio", "medium_ratio", "hard_ratio",
        "multi_chunk_ratio", "duplicate_ratio",
        "total_llm_calls", "total_tokens", "tokens_per_accepted",
    ]

    for label, this_key, prev_key in pairs:
        this = metrics.get(this_key)
        prev = metrics.get(prev_key)
        if this and prev:
            delta = {
                "label": label,
                "this": this_key,
                "prev": prev_key,
            }
            for k in numerical_keys:
                delta[k] = round(this.get(k, 0) - prev.get(k, 0), 4)
            deltas[label] = delta

    return deltas


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["run_id", "group_id", "seed", "accepted_count", "generated_count", "accept_rate",
            "easy_ratio", "medium_ratio", "hard_ratio",
            "multi_chunk_ratio", "duplicate_ratio",
            "total_llm_calls", "total_tokens", "tokens_per_accepted",
            "strategy_counts"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], deltas: dict[str, dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# QA-Agent 消融实验结果\n"]
    lines.append(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 各组指标表 ──
    lines.append("## 各组指标\n")
    cols = [
        ("run_id", "Run"),
        ("seed", "Seed"),
        ("accepted_count", "Accepted"),
        ("generated_count", "Generated"),
        ("accept_rate", "Accept Rate"),
        ("easy_ratio", "Easy%"),
        ("medium_ratio", "Medium%"),
        ("hard_ratio", "Hard%"),
        ("multi_chunk_ratio", "Multi-Chunk%"),
        ("total_llm_calls", "LLM Calls"),
        ("total_tokens", "Tokens"),
        ("tokens_per_accepted", "Tokens/Accepted"),
    ]
    header = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join(" --- " for _ in cols) + "|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        vals = []
        for k, _ in cols:
            v = row.get(k, "-")
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    # ── 策略分布 ──
    lines.append("\n## 策略分布\n")
    for row in rows:
        strategies = row.get("strategy_counts", {})
        lines.append(f"- **{row['run_id']}** (seed={row.get('seed', '-')}): {strategies}")

    # ── 消融对比 ──
    if deltas:
        lines.append("\n## 消融对比\n")
        lines.append("| 对比 | Accept Rate Δ | Easy Δ | Medium Δ | Hard Δ | Multi-Chunk Δ | Tokens Δ | Tokens/Accepted Δ |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for label, d in sorted(deltas.items()):
            lines.append(
                f"| {label} "
                f"| {d.get('accept_rate', 0):+.4f} "
                f"| {d.get('easy_ratio', 0):+.4f} "
                f"| {d.get('medium_ratio', 0):+.4f} "
                f"| {d.get('hard_ratio', 0):+.4f} "
                f"| {d.get('multi_chunk_ratio', 0):+.4f} "
                f"| {d.get('total_tokens', 0):+d} "
                f"| {d.get('tokens_per_accepted', 0):+.1f} |"
            )

        lines.append("\n### 论证\n")
        pairs = [
            ("B > A", "多轮生成有效", "B - A"),
            ("C > B", "反馈调度有效", "C - B"),
            ("D > C", "难度进化有效", "D - C"),
        ]
        for claim, hypothesis, delta_key in pairs:
            d = deltas.get(delta_key, {})
            acc_delta = d.get("accept_rate", 0)
            token_delta = d.get("tokens_per_accepted", 0)
            lines.append(
                f"- **{claim}** — {hypothesis}: "
                f"accept_rate {acc_delta:+.4f}, "
                f"tokens/accepted {token_delta:+.1f}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not RUNS_DIR.exists():
        print(f"ERROR: {RUNS_DIR} not found. 请先运行 run_all.py。")
        sys.exit(1)

    all_metrics: dict[str, dict] = {}

    # 遍历 seed_{seed}/{run_id}/ 目录结构
    for seed_dir in sorted(RUNS_DIR.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue

        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == SUMMARY_DIR.name:
                continue

            run_id = run_dir.name
            seed = seed_dir.name.replace("seed_", "")
            print(f"Processing: {seed_dir.name}/{run_id}")
            metrics = compute_metrics(run_dir)
            metrics["seed"] = seed
            # 用 run_id 聚合（同 run_id 不同 seed 取均值等处理）
            all_metrics[f"{run_id}_s{seed}"] = metrics

    if not all_metrics:
        print("No results found.")
        sys.exit(1)

    # ── 按 GROUP_ORDER 排序 ──
    rows = []
    for group in GROUP_ORDER:
        if group in all_metrics:
            rows.append(all_metrics[group])

    # ── 计算消融差值 ──
    deltas = compute_deltas(all_metrics)

    # ── 输出 ──
    write_csv(rows, SUMMARY_DIR / "ablation_summary.csv")
    write_md(rows, deltas, SUMMARY_DIR / "ablation_summary.md")

    print(f"\n✅ 汇总完成:")
    print(f"  CSV: {SUMMARY_DIR / 'ablation_summary.csv'}")
    print(f"  MD:  {SUMMARY_DIR / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
