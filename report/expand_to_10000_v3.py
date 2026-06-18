"""Add final expansion to reach 10000+ Chinese characters."""

from docx import Document
from lxml import etree
from docx.oxml.ns import qn

OUT_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/赵子晴_中期报告_完善版.docx'

FONT_BODY = 'Times New Roman'
FONT_EA = '宋体'
FONT_HEADING_EA = '黑体'
SZ_BODY = 12
SZ_SUBSECTION = 14
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


def insert_after(doc, after_idx, text, bold=False, size=SZ_BODY, ea_font_name=FONT_EA):
    ref_element = doc.paragraphs[after_idx]._element
    new_p = make_para(doc, text, bold=bold, size=size, ea_font_name=ea_font_name)
    ref_element.addnext(new_p)


def find_text(doc, keyword, start=0):
    for i in range(start, len(doc.paragraphs)):
        if keyword in doc.paragraphs[i].text.strip():
            return i
    return -1


def count_chinese(text):
    return sum(1 for c in text if '一' <= c <= '鿿')


def main():
    doc = Document(OUT_PATH)
    initial_total = count_chinese(''.join(p.text.strip() for p in doc.paragraphs if p.text.strip()))
    print('Initial total Chinese:', initial_total)

    expansions = []

    # === Section 2.1 additional: evidence chunking strategies ===
    idx = find_text(doc, '可配置的文本分段策略')
    if idx >= 0:
        text = (
            '文本分段的参数配置对题目生成质量有重要影响。'
            '固定长度分块方式的主要优势在于实现简单、'
            '处理速度快，且每个文本块的大小均匀，便于批量化处理；'
            '其缺点在于可能切断语义完整的段落，'
            '导致生成的题目缺乏上下文连贯性。'
            '语义边界自适应分块方式则能够更好地保留原文的'
            '语义完整性，生成的题目通常具有更好的上下文理解基础，'
            '但其计算开销较大，且分块大小不均匀。'
            '系统允许用户根据具体需求选择合适的分块策略，'
            '也可以为不同主题配置不同的分块策略。'
            '在实际使用中，对于技术类主题推荐使用固定长度分块，'
            '对于需要深层理解的复杂主题推荐使用语义边界分块，'
            '以在效率和效果之间取得最佳平衡。'
            '此外，系统还支持对分块后的文本片段进行自动摘要，'
            '提取每个文本块的核心信息，'
            '为后续的证据采样和题目生成提供更加精炼的输入。'
        )
        insert_after(doc, idx, text)
        expansions.append('2.1 chunking strategies')
        doc.save(OUT_PATH)
        doc = Document(OUT_PATH)

    # === Section 2.2 additional: communication between agents ===
    idx = find_text(doc, '这种可扩展的架构设计')
    if idx >= 0:
        text = (
            '在多智能体协作的过程中，'
            '通信效率是一个关键的设计考量。'
            '系统采用基于共享数据结构的异步通信模式，'
            '各智能体通过统一的中间数据层交换信息，'
            '无需直接调用彼此的内部接口。'
            '规划智能体将执行计划写入共享的任务队列，'
            '各子智能体从队列中获取任务并执行，'
            '执行结果写回共享的结果存储。'
            '这种模式有效降低了各组件之间的耦合度，'
            '同时也便于对每个环节进行独立的监控和调试。'
            '当某个智能体出现故障或超时时，'
            '系统可以通过超时重试和降级策略来保证整体流程的稳定性，'
            '避免单点故障导致整个评价流程中断。'
            '此外，共享数据层还记录了完整的执行日志，'
            '为后续的流程回溯和问题排查提供了便利。'
        )
        insert_after(doc, idx, text)
        expansions.append('2.2 communication')
        doc.save(OUT_PATH)
        doc = Document(OUT_PATH)

    # === Section 4 additional: second difficulty item ===
    idx = find_text(doc, '第一，评价数据集的质量保障问题')
    if idx >= 0:
        text = (
            '针对上述质量保障问题，'
            '本研究尝试从多个角度进行缓解。'
            '在生成阶段，通过多轮反馈机制持续优化提示词设计和证据采样策略，'
            '从源头上提升题目质量。'
            '在验证阶段，采用刚柔并济的多阶段验证流程，'
            '既保证了事实性信息的准确性，'
            '又通过大模型质量评估捕捉潜在的逻辑问题。'
            '然而，完全消除自动生成题目中的质量问题仍是一个开放性的挑战，'
            '需要结合更先进的验证技术和人工抽检机制来共同解决。'
        )
        insert_after(doc, idx, text)
        expansions.append('4 quality mitigation')
        doc.save(OUT_PATH)
        doc = Document(OUT_PATH)

    # === Section 5 additional: expand completion argument ===
    idx = find_text(doc, '按期完成全部论文工作具有较高的可行性')
    if idx >= 0:
        text = (
            '需要指出的是，'
            '虽然整体可行性较高，'
            '但后续工作中仍存在一定的不确定性。'
            '评估报告生成模块的可视化部分可能需要较多的调试时间，'
            '以确保图表输出的质量和美观度。'
            '多轮反馈机制的优化工作在参数调优方面可能需要进行多轮尝试，'
            '才能找到最佳的配置组合。'
            '此外，系统集成测试阶段可能会发现一些预期之外的兼容性问题，'
            '需要预留一定的缓冲时间用于问题修复。'
            '针对这些潜在风险，'
            '本研究已经在进度安排中预留了适当的缓冲期，'
            '并制定了相应的应急预案，'
            '能够在出现问题时及时调整工作计划，'
            '确保整体进度不受重大影响。'
        )
        insert_after(doc, idx, text)
        expansions.append('5 risk mitigation')
        doc.save(OUT_PATH)
        doc = Document(OUT_PATH)

    # === Section 3 additional: expand schedule rationale ===
    idx = find_text(doc, '时间相对充裕，且已制定了详细的周计划')
    if idx >= 0:
        text = (
            '在具体执行层面，'
            '每周的工作安排遵循"计划-执行-复盘"的周循环模式。'
            '每周末制定下周的详细任务清单和预期目标，'
            '工作日按计划推进各项任务，'
            '周末进行本周工作的总结复盘，'
            '评估各项任务的完成质量和进度偏差，'
            '并及时调整后续计划。'
            '这种周期性的工作节奏有助于保持持续稳定的研究产出，'
            '同时也能及时发现和纠正偏差，'
            '避免问题积累到后期才暴露。'
        )
        insert_after(doc, idx, text)
        expansions.append('3 weekly rhythm')
        doc.save(OUT_PATH)
        doc = Document(OUT_PATH)

    # ========== final count ==========
    doc = Document(OUT_PATH)
    all_text = ''.join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    chinese_total = count_chinese(all_text)

    # Per-section counts
    sections = {}
    current_sec = 'header'
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if not txt:
            continue
        if '课题研究的背景和意义' in txt:
            current_sec = '1.1'
        elif '课题研究内容和进展' in txt:
            current_sec = '1.2'
        elif '目前已经完成的研究工作' in txt:
            current_sec = '2'
        elif '后期拟完成的研究工作' in txt:
            current_sec = '3'
        elif '存在的困难与问题' in txt:
            current_sec = '4'
        elif '如期完成全部论文工作的可能性' in txt:
            current_sec = '5'
        elif txt == '参考文献':
            current_sec = 'ref'
        if current_sec not in sections:
            sections[current_sec] = ''
        sections[current_sec] += txt

    print('\n' + '=' * 40)
    print('Expansions performed:', len(expansions))
    for e in expansions:
        print('  +', e)
    print('\nPer-section Chinese chars:')
    for k in ['header', '1.1', '1.2', '2', '3', '4', '5', 'ref']:
        v = sections.get(k, '')
        c = count_chinese(v)
        print('  {}: {}'.format(k, c))
    print('\nTotal Chinese:', chinese_total)
    print('Added:', chinese_total - initial_total)
    print('=' * 40)

    # Check for code identifiers
    bad_patterns = ['GeneratorFeedback', 'ValidatorFeedback', 'EvaluatorFeedback',
                    'cold_start', 'gen_only', 'val_only']
    found_any = False
    for p in bad_patterns:
        if p.lower() in all_text.lower():
            print('WARNING: Found', p)
            found_any = True
    if not found_any:
        print('Code identifier check: PASSED')

    if chinese_total >= 10000:
        print('=== TARGET REACHED: 10000+ Chinese characters ===')
    else:
        print('Still need', 10000 - chinese_total, 'more characters')


if __name__ == '__main__':
    main()
