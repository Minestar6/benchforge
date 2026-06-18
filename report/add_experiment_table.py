"""Replace text-based experiment table with a proper Word table in section 2.1.2."""
from docx import Document
from lxml import etree
from docx.oxml.ns import qn, nsmap

OUT_PATH = '/Users/zhaoziqing/Desktop/benchforge/report/赵子晴_中期报告_完善版.docx'

FONT_BODY = 'Times New Roman'
FONT_EA = '宋体'
SZ_BODY = 12
ALIGN_CENTER = 'center'


def make_cell_para(text, bold=False, center=True):
    """Create a w:p element for table cell content."""
    new_p = etree.Element(qn('w:p'))
    pPr = etree.SubElement(new_p, qn('w:pPr'))
    if center:
        jc = etree.SubElement(pPr, qn('w:jc'))
        jc.set(qn('w:val'), 'center')
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
    if bold:
        etree.SubElement(rPr, qn('w:b'))
        etree.SubElement(rPr, qn('w:bCs'))
    t = etree.SubElement(r, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    return new_p


def make_cell(text, bold=False):
    """Create a w:tc element."""
    tc = etree.Element(qn('w:tc'))
    tcPr = etree.SubElement(tc, qn('w:tcPr'))
    tcW = etree.SubElement(tcPr, qn('w:tcW'))
    tcW.set(qn('w:w'), '1400')
    tcW.set(qn('w:type'), 'dxa')
    tc.append(make_cell_para(text, bold=bold))
    return tc


def make_row(cells, header=False):
    """Create a w:tr element. cells is a list of (text, bold) tuples."""
    tr = etree.Element(qn('w:tr'))
    trPr = etree.SubElement(tr, qn('w:trPr'))
    tblHeader = etree.SubElement(trPr, qn('w:tblHeader'))
    tblHeader.set(qn('w:val'), 'true')
    for text in cells:
        tr.append(make_cell(text, bold=header))
    return tr


def build_table():
    """Build the experiment result table."""
    tbl = etree.Element(qn('w:tbl'))

    # Table properties
    tblPr = etree.SubElement(tbl, qn('w:tblPr'))
    tblStyle = etree.SubElement(tblPr, qn('w:tblStyle'))
    tblStyle.set(qn('w:val'), 'TableGrid')
    tblW = etree.SubElement(tblPr, qn('w:tblW'))
    tblW.set(qn('w:w'), '5000')
    tblW.set(qn('w:type'), 'dxa')
    tblBorders = etree.SubElement(tblPr, qn('w:tblBorders'))
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = etree.SubElement(tblBorders, qn('w:' + border_name))
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')

    # Table grid (column widths)
    tblGrid = etree.SubElement(tbl, qn('w:tblGrid'))
    widths = [1800, 1200, 1100, 1100, 1000, 1000, 1000, 1000]
    for w in widths:
        gridCol = etree.SubElement(tblGrid, qn('w:gridCol'))
        gridCol.set(qn('w:w'), str(w))

    # Header row
    headers = ['组别', '模式', '原始数量', '最终入选', '通过率', '简单', '中等', '困难']
    tbl.append(make_row(headers, header=True))

    # Data rows
    data = [
        ['A组（直接基线）', '多选题', '75', '74', '98.7%', '68', '5', '1'],
        ['A组（直接基线）', '问答题', '80', '77', '96.3%', '71', '6', '0'],
        ['B组（多轮无反馈）', '多选题', '99', '82', '82.8%', '40', '39', '3'],
        ['B组（多轮无反馈）', '问答题', '101', '91', '90.1%', '21', '60', '10'],
        ['C组（反馈无难度进化）', '多选题', '80', '69', '86.3%', '15', '33', '21'],
        ['C组（反馈无难度进化）', '问答题', '79', '68', '86.1%', '20', '25', '23'],
        ['D组（完整方法）', '多选题', '84', '70', '83.3%', '12', '33', '25'],
        ['D组（完整方法）', '问答题', '81', '75', '92.6%', '22', '31', '22'],
    ]
    for row_data in data:
        tbl.append(make_row(row_data))

    return tbl


def find_text(doc, keyword):
    for i, p in enumerate(doc.paragraphs):
        if keyword in p.text.strip():
            return i
    return -1


def main():
    doc = Document(OUT_PATH)

    # Find the table caption paragraph
    caption_idx = find_text(doc, '表2.1 消融实验结果对比')
    if caption_idx < 0:
        print('ERROR: Could not find table caption')
        return
    print(f'Table caption at paragraph {caption_idx}')

    # Find the next section heading (2.2) to know where table data ends
    next_sec_idx = find_text(doc, '2.2 自动评价智能体架构')
    print(f'Next section at paragraph {next_sec_idx}')

    # Remove the 4 text-based table data paragraphs (they are at caption_idx+1, +2, +3, +4)
    # But also need to be careful: the analysis paragraphs start after these
    # Let's check what's between caption and 2.2
    for i in range(caption_idx + 1, next_sec_idx):
        txt = doc.paragraphs[i].text.strip()
        print(f'  [{i}] will remove: {txt[:60]}...')

    # Remove paragraphs (text table data + any between caption and 2.2)
    # Actually, we need to keep the analysis paragraphs (综合四组结果... and 需要指出的是...)
    # Let me be more precise about what to remove

    # Looking at the output:
    # [78] 表2.1 caption → keep
    # [79-82] text data → remove, replace with table
    # [83] 综合四组结果... → keep
    # [84] 需要指出的是... → keep

    # Find where analysis text starts
    analysis_idx = find_text(doc, '综合四组结果')
    print(f'Analysis starts at paragraph {analysis_idx}')

    # Remove paragraphs from caption_idx+1 to analysis_idx-1 (the text table data)
    elements_to_remove = []
    for i in range(caption_idx + 1, analysis_idx):
        elements_to_remove.append(doc.paragraphs[i]._element)
        print(f'  Removing paragraph {i}')

    for elem in elements_to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)

    print(f'Removed {len(elements_to_remove)} text table paragraphs')

    # Re-find caption index after removal
    doc.save(OUT_PATH)
    doc = Document(OUT_PATH)
    caption_idx = find_text(doc, '表2.1 消融实验结果对比')
    print(f'New caption index: {caption_idx}')

    # Insert the proper table after the caption
    ref_element = doc.paragraphs[caption_idx]._element
    table_elem = build_table()
    ref_element.addnext(table_elem)

    doc.save(OUT_PATH)
    print('Table inserted successfully')

    # Verify
    doc = Document(OUT_PATH)
    total_chinese = sum(1 for p in doc.paragraphs for c in p.text.strip() if '一' <= c <= '鿿')
    print(f'Total document: {total_chinese} Chinese characters')

    # Check no code identifiers
    all_text = ''.join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    bad_patterns = ['GeneratorFeedback', 'ValidatorFeedback', 'EvaluatorFeedback',
                    'cold_start', 'gen_only', 'val_only']
    for p in bad_patterns:
        if p.lower() in all_text.lower():
            print(f'WARNING: Found {p}')
            return
    print('Code identifier check: PASSED')


if __name__ == '__main__':
    main()
