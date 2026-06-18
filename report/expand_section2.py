"""Expand section 2 content to meet 3000 Chinese character requirement."""

from docx import Document
from lxml import etree
from docx.oxml.ns import qn

OUT_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/赵子晴_中期报告_完善版.docx'

FONT_BODY = 'Times New Roman'
FONT_EA = '宋体'
SZ_BODY = 12
ALIGN_JUSTIFY = 'both'


def make_para(doc, text, bold=False, size=SZ_BODY, font_name=FONT_BODY,
              ea_font_name=FONT_EA, alignment=ALIGN_JUSTIFY):
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


def insert_after(doc, after_idx, text):
    ref_element = doc.paragraphs[after_idx]._element
    new_p = make_para(doc, text)
    ref_element.addnext(new_p)


doc = Document(OUT_PATH)

# ---- Find insertion anchors ----
anchor_2_1_end = None
anchor_2_2_end = None
anchor_2_4_end = None

for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    if '2.2 规划智能体' in txt:
        anchor_2_1_end = i - 1
    if '2.3 题目验证智能体' in txt:
        anchor_2_2_end = i - 1
    if '2.5 评估报告生成' in txt:
        anchor_2_4_end = i - 1

print(f"2.1 end: {anchor_2_1_end}, 2.2 end: {anchor_2_2_end}, 2.4 end: {anchor_2_4_end}")

# --- 1. After 2.1: generation strategy details ---
gen_strategy_text = (
    '在生成策略方面，系统针对不同的题目模式设计了差异化的生成模板。'
    '对于问答题（QA），采用基于证据的问答生成策略，'
    '要求模型根据提供的证据片段生成问题及其标准答案，并标注答案在证据中的引用位置。'
    '对于多项选择题（Multiple Choice），'
    '采用基于证据的选项生成策略，在生成正确答案的同时构造具有迷惑性的干扰项，并确保所有选项均与证据内容相关。'
    '两种模式下均支持通过 few-shot 示例引导生成格式，保证输出结构的规范性和一致性。'
)
insert_after(doc, anchor_2_1_end, gen_strategy_text)
print("Added 2.1 expansion")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

# --- 2. After 2.2: diagnosis details ---
diagnosis_text = (
    '在诊断阶段，系统执行五种诊断路径的判别：冷启动（cold_start）表示系统首次运行或重置后无历史数据；'
    '仅生成（gen_only）表示当前轮次仅有生成反馈，尚未进入验证和评估阶段；'
    '验证-评估（val_eval）表示上一轮完成了完整的生成、验证和评估流程；'
    '生成-验证-评估（gen_val_eval）表示上一轮完成了全流程并获得了评估反馈；'
    '仅验证（val_only）表示直接对已有候选池进行验证和评估而不触发新生成。'
    '每条路径对应不同的参数更新策略，实现细粒度的自适应控制。'
)
# Re-find anchor after doc reload
for i, p in enumerate(doc.paragraphs):
    if '2.3 题目验证智能体' in p.text.strip():
        anchor_2_2_end = i - 1
        break
insert_after(doc, anchor_2_2_end, diagnosis_text)
print("Added 2.2 expansion")

doc.save(OUT_PATH)
doc = Document(OUT_PATH)

# --- 3. After 2.4: evaluation discriminative signals ---
discriminative_text = (
    '在区分度分析方面，报告提供了一系列关键量化指标：总体模型差距（overall_model_gap）衡量所有候选模型'
    '在全部题目上的平均分差；最佳与次佳模型差距（best_vs_second_gap）反映领先模型的优势程度；'
    '每个题目的方差（per_question_variance）分析揭示不同模型在各题目上的表现离散度；'
    '区分度题目比例（discriminative_question_ratio）标明能有效区分模型能力的题目占比；'
    '过易题目比例（easy_questions_too_easy_ratio）和全失败题目比例（all_models_fail_ratio）'
    '则为评价数据集的质量诊断提供依据，帮助识别需要优化的题目类型。'
)
for i, p in enumerate(doc.paragraphs):
    if '2.5 评估报告生成' in p.text.strip():
        anchor_2_4_end = i - 1
        break
insert_after(doc, anchor_2_4_end, discriminative_text)
print("Added 2.4 expansion")

doc.save(OUT_PATH)

# --- Verify ---
doc = Document(OUT_PATH)
sec2_text = ""
started = False
for i, p in enumerate(doc.paragraphs):
    txt = p.text.strip()
    # find "目前已经完成的研究工作" heading
    if i >= 60 and '目前已经完成的研究工作' in txt:
        started = True
        continue
    if started and '后期拟完成的研究工作' in txt:
        break
    if started and txt:
        sec2_text += txt

chinese = sum(1 for c in sec2_text if '一' <= c <= '鿿')
print(f"Section 2 content: {chinese} Chinese characters, {len(sec2_text)} total (requirement: >=3000)")

total = sum(1 for p in doc.paragraphs for c in p.text.strip() if '一' <= c <= '鿿')
print(f"Total: {total} Chinese characters, {len(doc.paragraphs)} paragraphs")
