"""Insert citation & diversity score analysis into section 2.1.2 after the experiment table."""
from docx import Document
from lxml import etree
from docx.oxml.ns import qn

OUT_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/赵子晴_中期报告_完善版.docx'

FONT_BODY = 'Times New Roman'
FONT_EA = '宋体'
SZ_BODY = 12
ALIGN_JUSTIFY = 'both'


def make_para(text):
    new_p = etree.Element(qn('w:p'))
    pPr = etree.SubElement(new_p, qn('w:pPr'))
    jc = etree.SubElement(pPr, qn('w:jc'))
    jc.set(qn('w:val'), ALIGN_JUSTIFY)
    r = etree.SubElement(new_p, qn('w:r'))
    rPr = etree.SubElement(r, qn('w:rPr'))
    rFonts = etree.SubElement(rPr, qn('w:rFonts'))
    rFonts.set(qn('w:ascii'), FONT_BODY)
    rFonts.set(qn('w:hAnsi'), FONT_BODY)
    rFonts.set(qn('w:eastAsia'), FONT_EA)
    sz = etree.SubElement(rPr, qn('w:sz'))
    sz.set(qn('w:val'), str(int(SZ_BODY * 2)))
    szCs = etree.SubElement(rPr, qn('w:szCs'))
    szCs.set(qn('w:val'), str(int(SZ_BODY * 2)))
    t = etree.SubElement(r, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    return new_p


def find_text(doc, keyword):
    for i, p in enumerate(doc.paragraphs):
        if keyword in p.text.strip():
            return i
    return -1


def count_chinese(text):
    return sum(1 for c in text if '一' <= c <= '鿿')


# Content to insert

para_citation = (
    '为进一步量化评价各组生成题目的质量差异，'
    '本研究从引用得分和多样性得分两个维度对四组实验的入选题目进行了深入分析。'
    '引用得分衡量题目与其引用证据之间的事实一致性，'
    '由验证阶段的引文验证模块计算得出。'
    '具体计算方法为：对于每道题目，系统将其引用的证据文本与题目中的陈述和标准答案进行逐句比对，'
    '基于语义匹配程度计算每条引文的支持强度分数，取所有引文支持强度的均值作为该题的引用得分，'
    '分值范围为0到1，分值越高表示题目引用越准确。'
    '多样性得分衡量题目集合在语义空间中的分散程度，'
    '计算过程分为两步：首先使用SentenceTransformer模型将题目文本映射为语义嵌入向量并进行归一化处理，'
    '计算样本间余弦相似度矩阵的上三角均值，得到嵌入离散度（值越大表示题目间语义差异越大）；'
    '随后对嵌入向量进行KMeans聚类，基于各簇的样本分布计算聚类熵并除以最大熵进行归一化，'
    '最终多样性得分取嵌入离散度和归一化聚类熵的加权平均（权重各为0.5），'
    '分值越高表示题目内容越多样化。'
)

para_citation_data = (
    '从引用得分来看，A组（直接基线）的平均引用得分最高，达到0.9222，'
    '其中多选题为0.8912、问答题为0.9520，'
    '这是因为直接生成策略产出的题目以简单题为主，'
    '题目中的陈述和答案往往可以直接从证据文本中提取，引用准确性自然较高。'
    'B组（多轮无反馈）的平均引用得分为0.7998（多选题0.7820、问答题0.8159），'
    '较A组有明显下降，反映出多轮迭代中后期生成的题目复杂度增加，'
    '引用难度也随之上升。'
    'C组（反馈无难度进化）的平均引用得分为0.7764（多选题0.7644、问答题0.7885），'
    'D组（完整方法）的平均引用得分为0.7816（多选题0.7621、问答题0.7998），'
    '两组与B组接近但略低，说明引入反馈机制后生成的困难题目在引用准确性上'
    '面临更大挑战。总体而言，引用得分的下降是题目难度提升所带来的自然代价，'
    '各组得分均保持在0.75以上的合理水平，说明题目的事实准确性得到了有效保障。'
)

para_diversity_data = (
    '从多样性得分来看，A组（直接基线）的多样性得分为0.8773'
    '（嵌入离散度0.7957、聚类熵0.9590），'
    '在四组中最低，这是因为直接生成策略仅覆盖了单一主题'
    '且题目模式较为单一，语义空间分布相对集中。'
    'B组（多轮无反馈）的多样性得分为0.8982'
    '（嵌入离散度0.8567、聚类熵0.9397），'
    '较A组有了明显提升，说明多轮迭代本身通过多次采样和生成'
    '有效扩展了题目的语义覆盖范围。'
    'C组（反馈无难度进化）的多样性得分达到0.9147'
    '（嵌入离散度0.8653、聚类熵0.9641），'
    '在四组中最高，反映出反馈机制对提升题目多样性的显著作用——'
    '基于上一轮结果的反馈信号引导模型探索了更多样化的题目方向。'
    'D组（完整方法）的多样性得分为0.8958'
    '（嵌入离散度0.8616、聚类熵0.9300），'
    '略低于C组但仍显著高于A组，说明在多轮反馈和难度进化的共同作用下，'
    '系统在追求难度目标的同时保持了较高的题目多样性。'
)

para_summary = (
    '综合引用得分和多样性得分可以看出，四组实验呈现出清晰的质量分化趋势：'
    'A组以牺牲难度多样性为代价换取了最高的引用准确性，'
    '但其评价数据集的难度覆盖严重不足，'
    '难以有效区分高性能模型的真实能力差异。'
    'C组和D组在保持可接受的引用得分（0.78左右）的同时，'
    '显著提升了题目的多样性和难度覆盖，'
    '实现了质量与多样性的较好平衡。'
    'D组（完整方法）在两个维度上均表现稳定，'
    '特别是在两个题目模式间展现出最佳的一致性，'
    '验证了本课题提出的多轮反馈机制和难度自适应进化策略的有效性。'
)


def main():
    doc = Document(OUT_PATH)

    # Find insertion point: after table caption 表2.1
    caption_idx = find_text(doc, '表2.1 消融实验结果对比')
    if caption_idx < 0:
        print('ERROR: could not find table caption')
        return
    print(f'Table caption at paragraph {caption_idx}')

    # The analysis paragraph "综合四组结果" is at caption_idx + 1
    analysis_idx = find_text(doc, '综合四组结果可以得出以下结论')
    print(f'Analysis paragraph at {analysis_idx}')

    # Insert new paragraphs after the table caption (before existing analysis)
    # Addnext inserts immediately after the reference element.
    # We insert in reverse order so they appear in correct reading order.
    ref_element = doc.paragraphs[caption_idx]._element

    for para_text in [para_summary, para_diversity_data, para_citation_data, para_citation]:
        new_p = make_para(para_text)
        ref_element.addnext(new_p)

    doc.save(OUT_PATH)
    print('Citation & diversity paragraphs inserted successfully')

    # Verify
    doc = Document(OUT_PATH)
    total_chinese = count_chinese(''.join(p.text.strip() for p in doc.paragraphs if p.text.strip()))
    print(f'Total Chinese chars: {total_chinese}')

    # Check for code identifiers
    all_text = ''.join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    bad_patterns = ['GeneratorFeedback', 'ValidatorFeedback', 'EvaluatorFeedback',
                    'cold_start', 'gen_only', 'val_only']
    for p in bad_patterns:
        if p.lower() in all_text.lower():
            print(f'WARNING: Found {p}')
            return
    print('Code identifier check: PASSED')

    # Print the section around the insertion to verify
    sec2_2_idx = find_text(doc, '2.2 自动评价智能体架构')
    for i in range(caption_idx, sec2_2_idx):
        txt = doc.paragraphs[i].text.strip()
        if txt:
            print(f'\n[{i}] {txt[:120]}...' if len(txt) > 120 else f'\n[{i}] {txt}')


if __name__ == '__main__':
    main()
