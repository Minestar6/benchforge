"""消融实验完整性检查：验证各组运行日志中的策略分布是否符合预期。"""

import json
import sys
from pathlib import Path
from collections import Counter


RUNS_DIR = Path("runs/qa_ablation")

# ── 各组允许和禁止的策略 ──
RULES = {
    "B_no_feedback": {
        "forbidden": {
            "HARD_GENERATE", "FOCUS_TOPIC", "FOCUS_DIFFICULTY", "EXPAND_EVIDENCE",
        },
        "must_mainly": "NORMAL_GENERATE",
        "fixed_difficulty": "medium",
    },
    "C_feedback_no_diff": {
        "forbidden": {
            "HARD_GENERATE", "FOCUS_DIFFICULTY",
        },
        "allowed": {
            "INITIAL_BREADTH", "FOCUS_TOPIC", "EXPAND_EVIDENCE", "NORMAL_GENERATE",
        },
    },
    "D_full": {
        "forbidden": set(),
        "allowed": {
            "HARD_GENERATE", "FOCUS_DIFFICULTY", "FOCUS_TOPIC",
            "EXPAND_EVIDENCE", "NORMAL_GENERATE", "INITIAL_BREADTH",
        },
    },
}


def load_round_summary(run_dir: Path) -> list[dict]:
    """加载 round_summary.jsonl，返回所有 round 记录列表。"""
    path = run_dir / "round_summary.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_group(run_id: str, run_dir: Path, rules: dict) -> list[str]:
    """检查单个 group 的策略分布，返回错误列表。"""
    errors: list[str] = []
    records = load_round_summary(run_dir)

    if not records:
        print(f"  [{run_id}] WARNING: no round_summary.jsonl found (可能是 direct generation，跳过)")
        return errors

    strategies = [r.get("strategy", "unknown") for r in records]
    difficulties = [r.get("difficulty", "unknown") for r in records]
    strategy_counts = Counter(strategies)
    difficulty_counts = Counter(difficulties)

    print(f"  [{run_id}] {len(records)} rounds")
    print(f"    strategies: {dict(strategy_counts)}")
    print(f"    difficulties: {dict(difficulty_counts)}")

    # 检查禁止策略
    forbidden = rules.get("forbidden", set())
    for s in forbidden:
        if strategy_counts.get(s, 0) > 0:
            msg = f"消融失败：出现了禁止策略 {s}（出现 {strategy_counts[s]} 次）"
            errors.append(msg)
            print(f"    ❌ {msg}")
        else:
            print(f"    ✓ 未出现禁止策略: {s}")

    # B 组特殊检查：固定难度
    fixed_difficulty = rules.get("fixed_difficulty")
    if fixed_difficulty:
        non_fixed = {d for d in difficulties if d != fixed_difficulty}
        if non_fixed:
            msg = f"消融失败：difficulty 应为 {fixed_difficulty}，但出现 {non_fixed}"
            errors.append(msg)
            print(f"    ❌ {msg}")
        else:
            print(f"    ✓ difficulty 固定为 {fixed_difficulty}")

    # B 组特殊检查：主要策略
    must_mainly = rules.get("must_mainly")
    if must_mainly and strategy_counts:
        top_strategy, top_count = strategy_counts.most_common(1)[0]
        if top_strategy != must_mainly:
            msg = f"主要策略应为 {must_mainly}，但主要是 {top_strategy} ({top_count}/{len(strategies)})"
            errors.append(msg)
            print(f"    ❌ {msg}")
        else:
            print(f"    ✓ 主要策略为 {must_mainly} ({top_count}/{len(strategies)})")

    return errors


def main():
    if not RUNS_DIR.exists():
        print(f"ERROR: {RUNS_DIR} not found. 请先运行 run_all.py。")
        sys.exit(1)

    all_errors: dict[str, list[str]] = {}
    groups_checked = 0

    for seed_dir in sorted(RUNS_DIR.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue

        for run_dir in sorted(seed_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_id = run_dir.name

            # 找到匹配的规则
            matched_rule = None
            for rule_key, rules in RULES.items():
                if run_id == rule_key:
                    matched_rule = rules
                    break

            if matched_rule is None:
                print(f"[{run_id}] 无匹配规则，跳过")
                continue

            print(f"\n{'─' * 60}")
            print(f"Checking: {seed_dir.name}/{run_id}")
            errors = check_group(run_id, run_dir, matched_rule)
            if errors:
                all_errors[f"{seed_dir.name}/{run_id}"] = errors
            groups_checked += 1

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"检查完成: {groups_checked} 组")
    if all_errors:
        print(f"❌ 发现问题 {sum(len(v) for v in all_errors.values())} 个:")
        for run_id, errors in all_errors.items():
            for e in errors:
                print(f"  [{run_id}] {e}")
        sys.exit(1)
    else:
        print("✅ 所有消融开关正确生效")
        sys.exit(0)


if __name__ == "__main__":
    main()
