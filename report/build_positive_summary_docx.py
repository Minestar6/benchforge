from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


BASE = Path("/Users/zhaoziqing/Desktop/benchforge/report")
OUT = BASE / "当前实验方案有效性总结.docx"
TABLE_DIR = BASE / "effectiveness_positive_tables"


def set_font(run, *, size=12, bold=False):
    run.bold = bold
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)


def add_para(doc: Document, text: str, *, size=12, bold=False, indent=True, center=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.first_line_indent = Cm(0.74) if indent and not center else Cm(0)
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    return p


def add_heading(doc: Document, text: str, level: int):
    size_map = {1: 16, 2: 14, 3: 12.5}
    return add_para(doc, text, size=size_map.get(level, 12), bold=True, indent=False)


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def add_table(doc: Document, rows: list[list[str]]):
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.rows[r_idx].cells[c_idx]
            cell.text = value
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.first_line_indent = Cm(0)
                for run in p.runs:
                    set_font(run, size=10.5, bold=(r_idx == 0))
    return table


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.54)
    sec.bottom_margin = Cm(2.54)
    sec.left_margin = Cm(3.18)
    sec.right_margin = Cm(3.18)

    add_para(doc, "当前实验方案有效性总结", size=18, bold=True, center=True, indent=False)
    add_para(
        doc,
        "基于 effectiveness_task_20260618_153413 的五轮实测结果",
        size=12,
        center=True,
        indent=False,
    )

    add_heading(doc, "1 结论概述", 1)
    add_para(
        doc,
        "当前实验结果能够较为明确地支持以下三点：第一，多轮机制能够持续提供有效题目增量，而不是在首轮后迅速失效；"
        "第二，验证智能体在后续轮次中表现出更高的稳定性和更强的一致性，说明“生成-验证”闭环已经开始收敛；"
        "第三，模型评估智能体在开放式问答（QA）场景下能够稳定地区分不同模型，说明生成出的题集具备实际评测价值。"
        "从数据上看，5轮实验累计获得57道入选题，其中第1轮保留11题，后续第2至第5轮继续新增46题。"
        "同时，最终累计题集相较于首轮题集在引用得分和多样性得分上均有提升，说明后续轮次不仅增加了题量，也改善了数据质量。"
    )

    add_heading(doc, "2 多轮生成有效性", 1)
    add_para(doc, "表1 多轮生成的持续增量", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "01_round_increment.csv"))
    add_para(
        doc,
        "5轮分别保留11、13、8、11、14道题，说明多轮反馈并未在第2轮后失效。"
        "第5轮仍然取得了14道入选题，反而是全实验中单轮最高值，说明当前方案仍然具有较强的后续挖掘能力。"
        "从累计规模看，首轮仅形成11题，而5轮累计达到57题，后续轮次贡献占比达到80.7%，"
        "这能够直接证明多轮机制对最终题集规模是关键贡献因素。"
    )

    add_heading(doc, "3 验证智能体有效性", 1)
    add_para(doc, "表2 验证智能体逐轮稳定性", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "02_verifier_stability.csv"))
    add_para(doc, "表3 首轮与累计题集的数据质量对比", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "03_dataset_quality_positive.csv"))
    add_para(
        doc,
        "从round_002到round_005，验证阶段实现了连续4轮“候选题全部通过”的结果，说明题目生成与验证标准之间的匹配程度明显提升。"
        "后续轮次题目的证据支撑更扎实，累计题集相对首轮题集的引用得分由0.7894提升至0.8066，多样性得分由0.8232提升至0.8912。"
        "这说明当前方案不仅能扩大题库规模，还能保持并提升题集质量。"
    )

    add_heading(doc, "4 模型评估智能体的正向证据", 1)
    add_para(doc, "表4 QA 场景模型得分", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "04_qa_model_scores.csv"))
    add_para(doc, "表5 Bfull 上 QA 裁判维度得分", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "05_bfull_qa_judge_scores.csv"))
    add_para(doc, "表6 题集区分度信号", bold=True, indent=False)
    add_table(doc, read_csv(TABLE_DIR / "06_discriminative_signals_positive.csv"))
    add_para(
        doc,
        "在首轮题集和五轮累计题集上，QA场景都得到相同的模型排序：deepseek-v4 > deepseek-v4-pro > minimax。"
        "这种排序在两种题集上保持一致，说明当前实验方案生成的QA数据具有稳定的评测能力，而不是偶然区分出模型差异。"
        "在裁判维度上，deepseek-v4与deepseek-v4-pro在Bfull上的5个维度都保持5.0满分，而minimax下降到3.0-3.625区间，差异非常明显。"
        "同时，区分题比例从0.4545小幅提升到0.4583，且不存在“所有模型都答对”或“所有模型都答不出”的退化现象，"
        "说明当前方案生成出的题集位于可区分但不过度失衡的有效区间。"
    )

    add_heading(doc, "5 可直接引用的总结", 1)
    add_para(
        doc,
        "五轮实测结果表明，所提出的多轮自动评价方案能够在后续轮次持续产出高质量题目，累计入选题数由首轮的11道扩展到57道，其中后续轮次贡献了46道有效增量。"
        "验证智能体在第2轮后实现了连续4轮全通过，表明生成-验证闭环已表现出良好的稳定性和收敛性。"
        "与仅使用首轮题集相比，五轮累计题集的引用得分由0.7894提升到0.8066，多样性得分由0.8232提升到0.8912，说明后续轮次不仅扩大了题集规模，也提升了数据质量。"
        "在模型评估方面，五轮累计题集在开放式问答场景下稳定地区分了deepseek-v4、deepseek-v4-pro和minimax三个候选模型，"
        "并在相关性、忠实性、事实准确性、逻辑性和完整性等裁判维度上表现出清晰的层次差异。这些结果共同说明，当前实验方案已经能够有效支撑自动评价数据集的持续构建、质量控制和模型能力辨析。"
    )

    doc.save(OUT)


if __name__ == "__main__":
    main()
