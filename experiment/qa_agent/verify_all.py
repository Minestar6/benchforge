"""verify_all.py：对 run_all.py 生成的所有组题目调用 verify_agent 做质量验证。

用法:
  cd /Users/zhaoziqing/Desktop/benchforge
  python experiment/qa_agent/verify_all.py

输出：每组原始题目数（按难度拆分）和最终通过验证的题目数（按难度拆分），
      同时生成 verify_all_report.json。
"""

import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

from benchforge.config.config import load_dotenv
from benchforge.models.loader import ModelLoader
from benchforge.agents.model_eval_agent.model_registry_loader import load_model_registry
from benchforge.agents.verify_agent.agent import run_verify_agent_from_shared_state
from benchforge.agents.verify_agent.config_loader import load_verify_agent_config

load_dotenv(PROJECT_ROOT / ".env")

MODEL_API_KEY = os.getenv("CUSTOM_API_KEY", "")
MODEL_BASE_URL = os.getenv("CUSTOM_API_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

EXPERIMENT_BASE = PROJECT_ROOT / "runs" / "ques_generate"
CONFIG_PATH = str(PROJECT_ROOT / "experiment" / "qa_agent" / "configs" / "verify_agent.yaml")
REGISTRY_PATH = str(PROJECT_ROOT / "config" / "model_registry.yaml")
TARGET_RUN_PREFIXES = [
    "A_direct_multiple_choice_",
    "A_direct_qa_",
    "B_no_feedback_multiple_choice_",
    "B_no_feedback_qa_",
    "C_feedback_no_diff_multiple_choice_",
    "C_feedback_no_diff_qa_",
    "D_full_multiple_choice_",
    "D_full_qa_",
]


def _latest_run_for_prefix(prefix: str) -> Path | None:
    candidates = [
        d for d in EXPERIMENT_BASE.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def resolve_experiment_dirs() -> list[Path]:
    latest_dirs = []
    for prefix in TARGET_RUN_PREFIXES:
        latest = _latest_run_for_prefix(prefix)
        if latest is not None:
            latest_dirs.append(latest)
    if latest_dirs:
        return latest_dirs
    return sorted([d for d in EXPERIMENT_BASE.iterdir() if d.is_dir()], key=lambda p: p.name)


def collect_orig_diff(run_dir: Path) -> Counter:
    """从 candidate_pool.json 统计原始难度分布。"""
    counter: Counter = Counter()
    for mode_dir in ["qa", "multiple_choice"]:
        cp = run_dir / mode_dir / "candidate_pool.json"
        if cp.exists():
            with open(cp, encoding="utf-8") as f:
                items = json.load(f)
            for item in items:
                diff = item.get("difficulty", item.get("estimated_difficulty", "unknown"))
                counter[str(diff)] += 1
    if counter:
        return counter

    # fallback: 旧版 Group A 只有 direct_questions.json
    dq = run_dir / "direct_questions.json"
    if dq.exists():
        with open(dq, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            diff = item.get("estimated_difficulty", item.get("difficulty", "unknown"))
            counter[str(diff)] += 1
    return counter


def _int_to_label(diff) -> str:
    """将难度值映射为 easy / medium / hard 标签。"""
    if isinstance(diff, (int, float)):
        if diff <= 3:
            return "easy"
        if diff <= 6:
            return "medium"
        return "hard"
    return str(diff).strip().lower()


def _load_orig_difficulty_map(run_dir: Path) -> dict[str, str]:
    """从 candidate_pool.json 加载原始难度映射 {question_id: label}。"""
    orig_map: dict[str, str] = {}
    for mode_dir in ["qa", "multiple_choice"]:
        cp = run_dir / mode_dir / "candidate_pool.json"
        if not cp.exists():
            continue
        with open(cp, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            qid = item.get("question_id", "")
            diff = item.get("difficulty", item.get("estimated_difficulty", "unknown"))
            orig_map[qid] = _int_to_label(diff)
    if orig_map:
        return orig_map

    # fallback: 旧版 Group A
    dq = run_dir / "direct_questions.json"
    if dq.exists():
        with open(dq, encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            qid = item.get("question_id", "")
            diff = item.get("estimated_difficulty", item.get("difficulty", "unknown"))
            orig_map[qid] = _int_to_label(diff)
    return orig_map


def collect_diff_corrections(
    validation_dir: Path,
    orig_difficulty_map: dict[str, str],
) -> dict:
    """统计 final_status=selected 的题目中难度标签被修正的情况。

    Returns:
        {
            "total_selected": int,
            "corrected_count": int,
            "correction_ratio": float,
            "transitions": {"easy→medium": 2, "medium→hard": 1, ...},
        }
    """
    vq_path = validation_dir / "validated_questions.jsonl"
    if not vq_path.exists():
        return {"total_selected": 0, "corrected_count": 0, "correction_ratio": 0.0, "transitions": {}}

    total = 0
    corrected = 0
    transitions: Counter = Counter()

    with open(vq_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("final_status") != "selected":
                continue
            total += 1

            cand = rec.get("candidate", {})
            qid = cand.get("question_id", "")
            orig_label = orig_difficulty_map.get(qid, "unknown")

            # 当前难度标签：优先用 llm_validation 的建议，回退到 candidate 字段
            llm_val = rec.get("llm_validation", {}) or {}
            suggested = llm_val.get("suggested_difficulty")
            if suggested in ("easy", "medium", "hard"):
                curr_label = suggested
            else:
                curr_label = _int_to_label(cand.get("estimated_difficulty", "unknown"))

            if orig_label != "unknown" and curr_label != orig_label:
                corrected += 1
                transitions[f"{orig_label}→{curr_label}"] += 1

    return {
        "total_selected": total,
        "corrected_count": corrected,
        "correction_ratio": corrected / total if total > 0 else 0.0,
        "transitions": dict(transitions),
    }


def collect_final_diff(validation_dir: Path) -> Counter:
    """从 validated_questions.jsonl 中统计 final_status=selected 的难度分布。

    优先使用 LLM 验证建议的 suggested_difficulty（当 difficulty_consistency 低分时），
    否则使用原始 estimated_difficulty。
    """
    counter: Counter = Counter()
    vq_path = validation_dir / "validated_questions.jsonl"
    if not vq_path.exists():
        return counter
    with open(vq_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("final_status") == "selected":
                # 优先使用 LLM 建议的难度标签
                llm_val = rec.get("llm_validation", {}) or {}
                suggested = llm_val.get("suggested_difficulty")
                if suggested in ("easy", "medium", "hard"):
                    counter[suggested] += 1
                    continue

                cand = rec.get("candidate", {})
                diff = cand.get("estimated_difficulty", "unknown")
                counter[_int_to_label(diff)] += 1
    return counter


async def main():
    if not MODEL_API_KEY:
        logger.error("CUSTOM_API_KEY not set in .env at {}", PROJECT_ROOT / ".env")
        sys.exit(1)

    # ── 日志文件 ──────────────────────────────────────────────────────────────────
    log_dir = EXPERIMENT_BASE / "verify_all_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"verify_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        encoding="utf-8",
    )
    logger.info(f"Log file: {log_path}")

    config = load_verify_agent_config(CONFIG_PATH)

    # 从 model_registry 解析 verify_agent 配置中的逻辑模型名
    registry = load_model_registry(REGISTRY_PATH)
    verify_model_name = config.llm_validation.model
    if verify_model_name not in registry:
        logger.error(
            "verify_agent.yaml llm_validation.model '{}' not found in model_registry. Available: {}",
            verify_model_name, list(registry.keys()),
        )
        sys.exit(1)
    verify_model_cfg = registry[verify_model_name]
    model_client = ModelLoader.load_model(verify_model_cfg)

    logger.info(
        "verify_agent model: registry='{}' → api='{}'",
        verify_model_name, verify_model_cfg.model_name,
    )

    orig_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)

    try:
        results: list[dict] = []
        dirs = resolve_experiment_dirs()

        for i, dir_path in enumerate(dirs, 1):
            ss_path = dir_path / "shared_state.json"
            if not ss_path.exists():
                logger.warning(f"No shared_state.json, skipping: {dir_path.name}")
                continue

            logger.info("=" * 60)
            logger.info(f"[{i}/{len(dirs)}] {dir_path.name}")
            logger.info("=" * 60)

            try:
                result = await run_verify_agent_from_shared_state(
                    shared_state_path=str(ss_path),
                    config=config,
                    model_client=model_client,
                )
                if result is None:
                    continue

                orig_diff = dict(collect_orig_diff(dir_path))
                validation_dir = dir_path / "validation"
                final_diff = dict(collect_final_diff(validation_dir))

                # 难度修正统计
                orig_difficulty_map = _load_orig_difficulty_map(dir_path)
                correction_stats = collect_diff_corrections(validation_dir, orig_difficulty_map)

                report_path = validation_dir / "validation_report.json"
                failed_by_stage = {}
                if report_path.exists():
                    with open(report_path, encoding="utf-8") as f:
                        rp = json.load(f)
                        failed_by_stage = rp.get("failed_by_stage", {})

                entry = {
                    "group": dir_path.name,
                    "orig_count": sum(orig_diff.values()),
                    "orig_by_diff": orig_diff,
                    "final_selected": len(result.selected_question_ids),
                    "final_by_diff": final_diff,
                    "failed_by_stage": failed_by_stage,
                    "diff_corrections": correction_stats,
                }
                results.append(entry)
                logger.info(f"  orig={entry['orig_count']} by_diff={orig_diff} → final={entry['final_selected']} by_diff={final_diff}")
                logger.info(
                    f"  diff_corrections: {correction_stats['corrected_count']}/{correction_stats['total_selected']} "
                    f"({correction_stats['correction_ratio']:.1%}) {correction_stats['transitions']}"
                )

            except Exception:
                logger.exception(f"Failed to verify {dir_path.name}")

    finally:
        os.chdir(orig_cwd)

    print_summary(results)

    report_out = EXPERIMENT_BASE / "verify_all_report.json"
    report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": verify_model_cfg.model_name,
            "registry_model": verify_model_name,
            "target_run_dirs": [str(p) for p in dirs],
            "config": {
                "min_citation_score": config.citation.min_citation_score,
                "min_overall_score": config.llm_validation.min_overall_score,
                "max_concurrency": config.llm_validation.max_concurrency,
                "alpha": config.citation.alpha,
                "beta": config.citation.beta,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {report_out}")
    logger.info(f"Log saved to {log_path}")


def print_summary(results: list[dict]):
    print("\n" + "=" * 90)
    print("  VERIFY ALL — Summary Report")
    print("=" * 90)

    total_orig = sum(r["orig_count"] for r in results)
    total_final = sum(r["final_selected"] for r in results)
    total_rate = (total_final / total_orig * 100) if total_orig > 0 else 0

    print(f"\n{'Group':<50} {'Original':>8} {'Final':>8} {'Pass Rate':>10}")
    print("-" * 80)

    for r in results:
        rate = (r["final_selected"] / r["orig_count"] * 100) if r["orig_count"] > 0 else 0
        print(f"{r['group']:<50} {r['orig_count']:>8} {r['final_selected']:>8} {rate:>9.1f}%")

    print("-" * 80)
    print(f"{'TOTAL':<50} {total_orig:>8} {total_final:>8} {total_rate:>9.1f}%")

    print(f"\n{'='*90}")
    print("  Per-Difficulty Breakdown")
    print("=" * 90)

    for r in results:
        print(f"\n── {r['group']} ──")
        print(f"  原始分布:  {r['orig_by_diff']}")
        print(f"  最终分布:  {r['final_by_diff']}")
        print(f"  各阶段剔除: {r.get('failed_by_stage', {})}")
        corr = r.get("diff_corrections", {})
        if corr.get("corrected_count", 0) > 0:
            print(f"  难度修正:    {corr['corrected_count']}/{corr['total_selected']} ({corr['correction_ratio']:.1%})")
            for trans, cnt in corr.get("transitions", {}).items():
                print(f"              {trans}: {cnt}")

    all_orig_diff: Counter = Counter()
    all_final_diff: Counter = Counter()
    all_failed: Counter = Counter()
    all_corrections: Counter = Counter()
    total_corrected = 0
    total_selected_for_corr = 0
    for r in results:
        for d, c in r["orig_by_diff"].items():
            all_orig_diff[d] += c
        for d, c in r["final_by_diff"].items():
            all_final_diff[d] += c
        for stage, c in r.get("failed_by_stage", {}).items():
            all_failed[stage] += c
        corr = r.get("diff_corrections", {})
        total_corrected += corr.get("corrected_count", 0)
        total_selected_for_corr += corr.get("total_selected", 0)
        for trans, cnt in corr.get("transitions", {}).items():
            all_corrections[trans] += cnt

    print(f"\n{'='*90}")
    print("  Overall Breakdown")
    print("=" * 90)
    print(f"  原始分布:  {dict(all_orig_diff)}")
    print(f"  最终分布:  {dict(all_final_diff)}")
    print(f"  各阶段剔除: {dict(all_failed)}")
    corr_rate = (total_corrected / total_selected_for_corr * 100) if total_selected_for_corr > 0 else 0
    print(f"  难度修正:    {total_corrected}/{total_selected_for_corr} ({corr_rate:.1f}%)")
    if all_corrections:
        print(f"  修正分布:    {dict(all_corrections)}")
    print(f"  总原始: {total_orig} → 总最终: {total_final} (通过率: {total_rate:.1f}%)\n")


if __name__ == "__main__":
    asyncio.run(main())
