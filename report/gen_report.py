"""生成中期报告内容并写入DOCX模板。"""

import copy, os, json
from pathlib import Path
from docx import Document
from docx.shared import Pt
from lxml import etree
from docx.oxml.ns import qn

DOC_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/模板_converted.docx'
OUT_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/赵子晴_中期报告_完善版.docx'

# ========== 样式常量 ==========
FONT_BODY = 'Times New Roman'
FONT_EA = '宋体'
FONT_HEADING_EA = '黑体'

SZ_BODY = 12        # 小4号
SZ_SUBSECTION = 14  # 4号
SZ_SECTION = 15     # 小3号

ALIGN_JUSTIFY = 'both'


# ========== XML 辅助函数 ==========
def make_paragraph(doc, text, bold=None, size=None, font_name=None, ea_font_name=None, alignment=None):
    new_p = etree.Element(qn('w:p'))

    pPr = etree.SubElement(new_p, qn('w:pPr'))
    if alignment is not None:
        jc = etree.SubElement(pPr, qn('w:jc'))
        jc.set(qn('w:val'), alignment)

    r = etree.SubElement(new_p, qn('w:r'))
    rPr = etree.SubElement(r, qn('w:rPr'))

    rFonts = etree.SubElement(rPr, qn('w:rFonts'))
    if font_name:
        rFonts.set(qn('w:ascii'), font_name)
        rFonts.set(qn('w:hAnsi'), font_name)
    if ea_font_name:
        rFonts.set(qn('w:eastAsia'), ea_font_name)

    if size:
        sz = etree.SubElement(rPr, qn('w:sz'))
        sz.set(qn('w:val'), str(int(size * 2)))
        szCs = etree.SubElement(rPr, qn('w:szCs'))
        szCs.set(qn('w:val'), str(int(size * 2)))

    if bold:
        etree.SubElement(rPr, qn('w:b'))
        etree.SubElement(rPr, qn('w:bCs'))

    t = etree.SubElement(r, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')

    return new_p


def insert_after(doc, after_idx, text, bold=None, size=None, font_name=None, ea_font_name=None, alignment=None):
    ref_element = doc.paragraphs[after_idx]._element
    new_p = make_paragraph(doc, text, bold, size, font_name, ea_font_name, alignment)
    ref_element.addnext(new_p)


def insert_section_heading(doc, after_idx, text):
    return insert_after(doc, after_idx, text, bold=True, size=SZ_SUBSECTION,
                        font_name=FONT_BODY, ea_font_name=FONT_HEADING_EA)


def insert_body(doc, after_idx, text):
    return insert_after(doc, after_idx, text, bold=False, size=SZ_BODY,
                        font_name=FONT_BODY, ea_font_name=FONT_EA,
                        alignment=ALIGN_JUSTIFY)


def insert_main_heading(doc, after_idx, text):
    return insert_after(doc, after_idx, text, bold=True, size=SZ_SECTION,
                        font_name=FONT_BODY, ea_font_name=FONT_HEADING_EA)


# ========== 主流程 ==========
doc = Document(DOC_PATH)

# ---- 定位关键位置 ----
idx_1_2_end = -1
idx_sec2 = -1
idx_2_1_end = -1

for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    if '评价数据集构造主要目标为' in txt:
        idx_1_2_end = i
    if '目前已经完成的研究工作' in txt:
        idx_sec2 = i
    if '因此，本研究提出的评价数据集构造方法' in txt:
        idx_2_1_end = i

print(f"idx_1_2_end={idx_1_2_end}, idx_sec2={idx_sec2}, idx_2_1_end={idx_2_1_end}")

# ====================
# 1. 在 1.2 末尾补充论文结构说明
# ====================
insert_body(doc, idx_1_2_end,
    "本论文围绕上述三个子任务，共分为四章展开：第2章详细介绍自动评价数据集构建方法（题目生成智能体），"
    "包括基于多轮反馈的自适应生成机制；第3章详细介绍大模型自动评价智能体架构，涵盖规划智能体、题目验证智能体和"
    "模型评估智能体三个核心组件；第4章介绍评估报告生成方法，基于多维度评估结果自动生成综合评价报告。"
    "其中，第2章和第3章已基本完成实现，第4章正在规划中。")
print("1. 补充第1章论文结构说明")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)


# ====================
# 2.1 评价数据集构造 - 详细展开
# ====================
idx_2_1_end = -1
for i, p in enumerate(doc.paragraphs):
    if '本研究提出的评价数据集构造方法' in p.text:
        idx_2_1_end = i

insert_body(doc, idx_2_1_end,
    "具体而言，题目生成智能体采用按模式分阶段的生成策略，当前支持问答题（QA）和多项选择题（Multiple Choice）两种模式。"
    "每种模式下，系统根据用户指定的主题（topic）和目标数量，自动进行相关资料检索与文本分段，构建证据库（EvidenceManager）。"
    "生成器（Generator）基于检索到的证据片段，结合精心设计的提示词模板，调用大模型生成候选题目。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_find = -1
for i, p in enumerate(doc.paragraphs):
    if '本研究提出的评价数据集构造方法' in p.text:
        idx_find = i

insert_body(doc, idx_find + 1,
    "题目生成智能体的核心创新在于多轮反馈自适应机制。每一轮生成完成后，系统对生成结果进行评估，收集生成反馈"
    "（GeneratorFeedback），包括本轮生成数量、难度分布、连续空轮次数等信息。基于这些反馈，系统动态调整下一轮的"
    "生成策略，包括：调整生成难度分布（从 INITIAL_BREADTH 初始广度覆盖，到 NORMAL_GENERATE 常规生成，再到 "
    "HARD_GENERATE 困难题目生成，直至 EXPAND_EVIDENCE 扩展证据和 TERMINAL_HARD_REPAIR 终末困难修复）；"
    "自适应调整证据块采样策略，优先选择尚未充分利用的证据片段；"
    "以及通过产量估计（yield estimation）预判当前资源的生成潜力，避免无效生成。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_find = -1
for i, p in enumerate(doc.paragraphs):
    if '自适应调整证据块采样策略' in p.text:
        idx_find = i

insert_body(doc, idx_find,
    "在证据处理方面，系统支持多种文本分段策略，包括基于滑动窗口的固定长度分块和基于语义边界的自适应分块。"
    "每个证据块附带来源元数据（如页码、章节标题），确保生成的题目具有良好的可追溯性。生成过程中，系统使用"
    "预分配-保留锁机制（pre-sampling with reservation locks），在多主题并行生成时公平分配证据资源，避免主题间资源竞争。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_find = -1
for i, p in enumerate(doc.paragraphs):
    if '预分配-保留锁机制' in p.text:
        idx_find = i

insert_body(doc, idx_find,
    "此外，生成过程受到多种停止条件的约束，包括：候选池达到目标数量上限、达到最大生成轮数、连续空轮次数超过阈值、"
    "以及生成失败率达到上限。通过这些机制，题目生成智能体能够在不依赖人工干预的情况下，自主完成高质量评价数据集的构建。"
    "目前，该组件已完整实现并通过多组实验验证了其有效性。")
print("2. 展开 2.1 评价数据集构造")


# ====================
# 2.2 规划智能体
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '已完整实现并通过多组实验验证了其有效性' in p.text:
        idx_anchor = i

insert_section_heading(doc, idx_anchor, "2.2 规划智能体")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '2.2 规划智能体' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "规划智能体（Planner Agent）是整个自动评价系统的核心编排器，负责将用户意图转化为可执行的评价计划，"
    "并在多轮执行过程中持续监控和调整。其核心工作流程为诊断-规划-执行-反馈的闭环迭代过程。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '核心编排器' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "在每轮迭代开始时，规划智能体首先执行诊断阶段（Diagnose），分析上一轮的生成反馈（GeneratorFeedback）、"
    "验证反馈（ValidatorFeedback）和评估反馈（EvaluatorFeedback），产生诊断标签（如 cold_start、gen_only、"
    "val_only、val_eval、gen_val_eval 等）。根据诊断结果，规划智能体分别生成三个维度的计划：题目计划（question_plan）"
    "确定本轮需要生成的题目数量、难度分布和主题；控制计划（control_plan）配置验证阈值、选择模式和评估配置；"
    "主题自适应检索（topic_adaptive_retriever）动态选择需要重点关注的主题。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '主题自适应检索' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "在难度演化方面，规划智能体实现了两种互补机制：基于规则的难度演化器（question_difficulty_evolver）根据诊断标签进行"
    "规则化的难度和数量调整，例如冷启动阶段优先铺量覆盖、困难不足时增加困难题比例；基于大模型的难度演化器"
    "（_llm_question_difficulty_evolver）利用大模型的语义理解能力，综合分析历史生成数据和当前需求，自动输出优化后的"
    "难度分布和目标数量。在参数调优方面，控制参数调优器（control_parameter_tuner）根据验证和评估反馈，"
    "自动调整引文验证阈值、选择模式（如 citation_only、citation_llm）和评估配置档案（eval_profile），"
    "实现全链路的自适应优化。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '两种互补机制' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "规划智能体还负责将上述计划组装为可执行的轮次规范（RoundSpec），并通过编排器（Orchestrator）执行。"
    "编排器负责合并各组件的基准配置与补丁配置，生成有效配置（effective config），依次调用题目生成智能体、"
    "验证智能体和模型评估智能体，并收集各阶段的反馈信息，形成完整的闭环。")
print("3. 完成 2.2 规划智能体")


# ====================
# 2.3 题目验证智能体
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '形成完整的闭环' in p.text:
        idx_anchor = i

insert_section_heading(doc, idx_anchor, "2.3 题目验证智能体")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '2.3 题目验证智能体' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "题目验证智能体（Verify Agent）负责对生成候选题目进行多阶段质量验证和筛选，确保最终纳入评价集的题目具有高质量、"
    "多样性和可评价性。验证流程分为三个阶段：引文验证、大模型质量评估和加权选择。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '引文验证、大模型质量评估和加权选择' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第一阶段为引文验证（Citation Validation），对每道题目的引用来源进行逐条验证，检查引文内容是否真实支持题目中的"
    "陈述和答案。验证结果记录为 structured citations，包含支持强度评分和支持原文引用。同时在这一阶段进行精确去重"
    "（Exact Dedup），基于题目文本的哈希值移除完全重复的题目。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '精确去重' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第二阶段为大模型质量评估（LLM Validation），调用大模型对候选题目进行多维度的质量评分，包括题目清晰度、"
    "答案正确性、引文一致性、难度合理性等方面。在进入此阶段前，系统还执行语义近似去重（Semantic Near-dedup），"
    "通过计算题目嵌入向量的余弦相似度，移除语义高度重复的题目，保证评价集的多样性。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '语义近似去重' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第三阶段为加权选择（Weighted Selection），根据引文验证分数和大模型质量评分计算每道题目的综合权重，"
    "按（模式, 难度）分组进行等比例采样（proportional sampling），确保最终选择的题目在覆盖全部主题的同时，"
    "各模式和难度层级保持合理的分布。最终输出的 validated_questions.jsonl 记录了每道题目的完整验证信息和最终状态"
    "（selected/rejected），为后续评估提供高质量的数据基础。")
print("4. 完成 2.3 验证智能体")


# ====================
# 2.4 模型评估智能体
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '高质量的数据基础' in p.text:
        idx_anchor = i

insert_section_heading(doc, idx_anchor, "2.4 模型评估智能体")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '2.4 模型评估智能体' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "模型评估智能体（Model Evaluation Agent）负责对候选大模型在构建的评价数据集上进行系统性评估，"
    "自动生成多维度、多粒度的评估报告。评估流程分为四个主要步骤：数据集指标计算、候选模型推理、"
    "自动指标评分和 LLM Judge 评估。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '数据集指标计算、候选模型推理' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第一步，数据集指标计算（Dataset Metrics）：对构建的评价数据集本身进行质量评估，包括引文覆盖率"
    "（Citation Score），衡量题目答案是否有可靠的引文支持；多样性分数（Diversity Score），通过嵌入向量的"
    "离散度和聚类熵评估题目的覆盖广度。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '引文覆盖率' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第二步，候选模型推理（Model Inference）：将评价集中的题目逐一输入待评估的候选大模型，收集模型生成的回答。"
    "支持多种模型的并行推理，通过统一的模型客户端接口（BaseModelClient）进行抽象，兼容不同厂商和类型的模型。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '候选模型推理（Model Inference）' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第三步，自动指标评分（Automatic Metrics）：对模型回答进行自动化的客观指标评估。支持的指标包括：精确匹配"
    "（Exact Match）、关键词匹配（Keyword Match）、语义相似度（Semantic Similarity）等，每种模式可以配置不同的"
    "指标组合和权重。")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '自动指标评分（Automatic Metrics）' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "第四步，LLM Judge 评估：调用大模型作为裁判（Judge），对模型回答进行更深入的质性评估。Judge 模型根据配置的"
    "评估维度和评分标准，对回答的正确性、完整性、逻辑性等进行综合打分。最终，系统通过综合评估报告生成器"
    "（build_comprehensive_eval_report）生成结构化的评估报告，包含按模式（Mode）、按主题（Topic）、"
    "按难度（Difficulty）和按模型（Model）的多维度指标分解，以及区分度分析（Discriminative Signals），"
    "包括模型间差距、每个题目的方差等信息，为模型能力分析提供丰富的数据支撑。")
print("5. 完成 2.4 模型评估智能体")


# ====================
# 2.5 评估报告生成（尚未实现）
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '数据支撑' in p.text:
        idx_anchor = i

insert_section_heading(doc, idx_anchor, "2.5 评估报告生成")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '2.5 评估报告生成' in p.text:
        idx_anchor = i

insert_body(doc, idx_anchor,
    "评估报告生成是本研究的重要组成部分，目标是将模型评估智能体产生的结构化评估数据，转化为直观、可读的综合评估报告。"
    "目前，系统已实现了评估报告的 JSON 格式输出（evaluation_report.json），包含按模式、主题、难度和模型的多维度指标分解，"
    "以及区分度分析信号。在此基础上，计划进一步开发面向人类的可读报告生成能力，包括自动生成评估报告的文本描述、"
    "可视化图表（如雷达图、柱状图、热力图等），以及基于自然语言的分析结论和建议。该部分功能目前处于规划设计阶段，"
    "将在后续工作中完成。")
print("6. 完成 2.5 评估报告生成")


# ====================
# 3. 后期拟完成的研究工作及进度安排
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '将在后续工作中完成' in p.text:
        idx_anchor = i

insert_main_heading(doc, idx_anchor, "3．后期拟完成的研究工作及进度安排")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('3．后期拟完成的研究工作'):
        idx_anchor = i

insert_body(doc, idx_anchor, "根据当前研究进度，后续工作按周安排如下：")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

schedule_items = [
    "第1-2周（2026.06.23 - 2026.07.06）：完成评估报告生成模块的开发。实现基于评估报告 JSON 的可视化图表生成功能，包括雷达图、柱状图等；开发报告文本描述自动生成模块，输出自然语言的分析结论。",
    "第3-4周（2026.07.07 - 2026.07.20）：完善多轮反馈自适应机制。针对当前实现中的不足，优化规划智能体的诊断策略和难度演化算法；进行多轮生成的稳定性测试和调优。",
    "第5-6周（2026.07.21 - 2026.08.03）：开展系统集成测试和实验验证。设计对比实验方案，验证本系统相对于传统评价方法的效率和效果优势；收集实验数据，分析各组件性能。",
    "第7-8周（2026.08.04 - 2026.08.17）：撰写论文初稿。完成硕士学位论文的初稿撰写，重点包括第2-4章的技术细节和实验分析。",
    "第9-10周（2026.08.18 - 2026.08.31）：论文修改和完善。根据导师意见修改论文，补充实验数据，完善参考文献，完成论文终稿。",
    "第11周（2026.09.01 - 2026.09.07）：准备答辩材料，进行论文答辩。",
]

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '根据当前研究进度' in p.text:
        idx_anchor = i
        break

for item in schedule_items:
    insert_body(doc, idx_anchor, item)
    doc.save(OUT_PATH)
    doc = Document(OUT_PATH)
    for i, p in enumerate(doc.paragraphs):
        if item[:20] in p.text:
            idx_anchor = i
            break

print("7. 完成第3节 进度安排")


# ====================
# 4. 存在的困难与问题
# ====================
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '论文答辩' in p.text:
        idx_anchor = i

insert_main_heading(doc, idx_anchor, "4．存在的困难与问题")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('4．存在的困难与问题'):
        idx_anchor = i

insert_body(doc, idx_anchor, "目前研究中存在以下技术难点和挑战：")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

difficulties = [
    "第一，评价数据集的质量保障问题。自动生成的题目在质量上仍难以完全达到人工构建的水平，特别是在复杂推理题目"
    "和领域专业题目的生成上，大模型可能出现事实性错误或逻辑不一致。虽然验证智能体能够过滤大部分低质量题目，"
    "但仍可能遗漏一些表面合格但实际存在问题的题目。如何进一步提升生成和验证的质量是一个持续的挑战。",
    "第二，多轮反馈机制的收敛性问题。在多轮生成过程中，反馈信号的准确性和时效性直接影响后续轮次的优化效果。"
    "当前实现中存在反馈延迟和信号噪声问题，可能导致生成策略的震荡或不收敛。需要研究更加鲁棒的反馈聚合策略和"
    "自适应学习率调节机制。",
    "第三，计算资源消耗问题。完整的自动评价流程涉及多个大模型的调用，包括题目生成、验证、评估和裁判等多个环节，"
    "计算成本较高。特别是在多轮迭代场景下，资源消耗会成倍增加。如何在保证评价质量的前提下优化资源利用效率"
    "是一个需要解决的实际问题。",
]

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '目前研究中存在' in p.text:
        idx_anchor = i
        break

for item in difficulties:
    insert_body(doc, idx_anchor, item)
    doc.save(OUT_PATH)
    doc = Document(OUT_PATH)
    for i, p in enumerate(doc.paragraphs):
        if item[:20] in p.text:
            idx_anchor = i
            break

print("8. 完成第4节 困难与问题")


# ====================
# 5. 如期完成全部论文工作的可能性
# ====================
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if '需要解决的实际问题' in p.text:
        idx_anchor = i

insert_main_heading(doc, idx_anchor, "5．如期完成全部论文工作的可能性")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('5．如期完成全部论文工作的可能性'):
        idx_anchor = i

insert_body(doc, idx_anchor,
    "总体而言，课题的核心技术框架已经完成，题目生成智能体、规划智能体、验证智能体和模型评估智能体均已实现，"
    "系统的基本流程已经跑通。剩余的工作主要集中在评估报告生成模块的开发、系统优化和实验验证上，这些工作基于"
    "现有框架进行，技术路线清晰，实现难度相对可控。按照当前的进度安排，预计能够如期完成全部论文工作，"
    "在2026年9月初提交论文终稿并参加答辩。")
print("9. 完成第5节 完成可能性")


# ====================
# 参考文献
# ====================
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

idx_anchor = len(doc.paragraphs) - 1
for i in range(len(doc.paragraphs) - 1, 0, -1):
    if doc.paragraphs[i].text.strip():
        idx_anchor = i
        break

insert_main_heading(doc, idx_anchor, "参考文献")
doc.save(OUT_PATH)
doc = Document(OUT_PATH)

refs = [
    "[1] Wang L, Ma C, Feng X, et al. A survey on large language model based autonomous agents[J]. Frontiers of Computer Science, 2024, 18(6): 186345.",
    "[2] CrewAI. CrewAI: Framework for orchestrating autonomous AI agents[EB/OL]. https://github.com/crewAIInc/crewAI, 2024.",
    "[3] Guo Z, Jin R, Liu C, et al. Evaluating large language models: A comprehensive survey[J]. arXiv preprint arXiv:2310.19736, 2023.",
    "[4] Chang Y, Wang X, Wang J, et al. A survey on evaluation of large language models[J]. ACM Transactions on Intelligent Systems and Technology, 2024, 15(3): 1-45.",
    "[5] Liang P, Bommasani R, Lee T, et al. Holistic evaluation of language models[J]. Annals of the New York Academy of Sciences, 2023, 1525(1): 140-146.",
    "[6] Gao L, Tow J, Abbasi B, et al. A framework for few-shot language model evaluation[EB/OL]. https://github.com/EleutherAI/lm-evaluation-harness, 2023.",
    "[7] Zheng L, Chiang W L, Sheng Y, et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena[C]//Advances in Neural Information Processing Systems, 2023.",
    "[8] Li B, Qi B, Liu T, et al. Dynamic evaluation of large language models by meta assessing agents[J]. arXiv preprint arXiv:2406.14889, 2024.",
    "[9] Liu X, Yu P, Zhang H, et al. Dynamic benchmarking framework for LLMs[J]. arXiv preprint arXiv:2402.00921, 2024.",
]

idx_anchor = -1
for i, p in enumerate(doc.paragraphs):
    if p.text.strip() == '参考文献':
        idx_anchor = i
        break

for ref in refs:
    insert_body(doc, idx_anchor, ref)
    doc.save(OUT_PATH)
    doc = Document(OUT_PATH)
    for i, p in enumerate(doc.paragraphs):
        if ref[:20] in p.text:
            idx_anchor = i
            break

print("10. 完成参考文献")


# ========== 最终保存 ==========
doc.save(OUT_PATH)

final_doc = Document(OUT_PATH)
print(f"\n总段落数: {len(final_doc.paragraphs)}")
print("\n内容结构:")
for i, p in enumerate(final_doc.paragraphs):
    txt = p.text.strip()
    if txt and (txt[0].isdigit() and ('．' in txt or '.' in txt or '．' in txt)):
        print(f"  [{i}] {txt[:80]}")

print(f"\n报告已保存至: {OUT_PATH}")
