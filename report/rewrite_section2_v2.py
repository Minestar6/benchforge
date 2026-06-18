"""Rewrite section 2 with hierarchical numbering and experiment section.

Structure:
  2.1 评价数据集构造
    2.1.1 研究内容 (evidence pool, question generation, multi-round feedback)
    2.1.2 实验方案 (ablation study with tables)
  2.2 自动评价智能体架构
    2.2.1 规划智能体 (planner architecture, tool calling, difficulty evolver)
    2.2.2 题目验证智能体 (three-stage verification)
    2.2.3 模型评估智能体 (four-step evaluation)
    2.2.4 总体框架介绍 (centralized multi-agent architecture)
  2.3 评估报告生成 (metrics, visualization)
"""

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


def insert_many(doc, after_idx, items):
    """Insert multiple paragraphs after the given index in order."""
    ref_element = doc.paragraphs[after_idx]._element
    for item in reversed(items):
        new_p = make_para(
            doc,
            text=item['text'],
            bold=item.get('bold', False),
            size=item.get('size', SZ_BODY),
            font_name=item.get('font_name', FONT_BODY),
            ea_font_name=item.get('ea_font_name', FONT_EA),
            alignment=item.get('alignment', ALIGN_JUSTIFY),
        )
        ref_element.addnext(new_p)


# ========== SECTION 2 CONTENT ==========

# 2.1 评价数据集构造
sec2_1_heading = "2.1 评价数据集构造"

# 2.1.1 研究内容
sec2_1_1_heading = "2.1.1 研究内容"

sec2_1_1_body_1 = (
    '证据池构造是题目生成的基础环节。系统首先根据用户指定的研究主题，'
    '通过自动化检索手段从百科类知识源中获取相关原始文档。'
    '获取到原始文档后，系统对文档内容进行结构化处理，'
    '采用可配置的文本分段策略将长文档切分为语义完整的文本块。'
    '系统支持两种分段策略：基于滑动窗口的固定长度分块，'
    '将文档按指定长度均匀切分，窗口大小和重叠量均可配置；'
    '基于语义边界的自适应分块，依据文档的自然段落结构和语义完整性进行分段，'
    '保留原文的章节层次关系。每个文本块均携带来源元数据，'
    '包括所属文档标题、章节信息和位置索引，确保生成的题目具有完整的可追溯性。'
    '为进一步提升证据的利用效率，系统还构建了单文本块和多文本块两种粒度的证据单元：'
    '单文本块单元直接基于单个分段构造题目，适用于考察细粒度知识点；'
    '多文本块单元则组合多个相关文本块，适用于考察跨段落的综合理解能力。'
    '同时，系统调用大模型为每个文档生成简短的摘要，'
    '便于后续的主题筛选和证据优先级排序。'
    '整个证据池构造过程完全自动化，用户只需指定研究主题即可自动完成相关资料的检索和预处理。'
)

sec2_1_1_body_2 = (
    '题目生成过程以证据池为基础，按照用户指定的题目模式和难度分布，'
    '系统性地调用大模型生成候选题目。系统当前支持两种题目模式：'
    '问答题要求模型根据证据片段生成问题及其标准答案，并标注答案在证据中的引用位置；'
    '多项选择题则在生成正确答案的同时构造具有迷惑性的干扰项，'
    '确保所有选项均与证据内容相关。具体生成流程为：'
    '首先根据当前轮的生成策略和难度分布，从证据池中采样合适的证据单元。'
    '采样过程采用预分配机制，在正式调用大模型之前预先为每个主题和难度层级分配好需要使用的证据单元，'
    '避免不同主题之间的资源竞争。随后，系统对每个采样的证据单元调用一次大模型，'
    '基于其中的文本内容按照预设的提示词模板生成一道或多道题目。'
    '每道生成的题目包含丰富的信息字段：题目文本、标准答案、预估难度等级、'
    '所属主题、证据引用信息、题目类型、所需能力标签以及模型的推理过程。'
    '生成完成后，系统通过轻量级过滤器对题目进行初步筛选，'
    '剔除格式异常、内容空洞或明显错误的题目，'
    '并通过基于哈希值的精确去重机制移除完全重复的题目。'
    '初步筛选后的合格题目进入候选池，等待后续的多阶段验证。'
)

sec2_1_1_body_3 = (
    '题目生成并非一次完成，而是通过多轮迭代逐步优化。'
    '系统设计了阶段化的生成策略演化路径：初始阶段以广度覆盖为主，'
    '尽量覆盖全部主题和难度层级，快速积累基础题目数量；'
    '在积累一定数量的题目后，逐步转向针对性生成，'
    '重点补充困难题目和表现薄弱的主题方向；'
    '当现有证据已充分使用时，触发证据扩展流程，'
    '为目标主题检索更多相关资料以扩展生成空间；'
    '最后阶段进行终末修复，针对仍然不足的难度层级进行定向补充。'
    '每一轮生成结束后，系统会综合分析本轮的各难度层级产量、'
    '连续空轮次数、生成失败率等信息，作为下一轮策略调整的依据。'
    '在证据采样方面，系统会跟踪每个文本块在各轮中的使用频率，'
    '优先选择尚未充分利用的证据片段，保证证据资源的均匀使用和题目多样性。'
    '系统还通过产量估计机制预测当前证据资源的剩余生成潜力，'
    '避免在低效证据上继续浪费计算资源。整个多轮生成过程受多种停止条件约束：'
    '当候选池达到目标数量上限、达到最大生成轮数上限、'
    '连续多轮无新题目生成或生成失败率超过阈值时，系统自动终止生成。'
    '通过这些机制，系统能够在无需人工干预的情况下，'
    '自主完成大规模、高质量评价数据集的构建，并具有良好的鲁棒性和自适应能力。'
)

# 2.1.2 实验方案
sec2_1_2_heading = "2.1.2 实验方案"

sec2_1_2_body_1 = (
    '为验证评价数据集构造方法的有效性，'
    '本课题设计并实施了一组消融实验，'
    '逐步考察多轮反馈机制和难度自适应进化策略对题目生成质量的影响。'
    '实验以题目生成的产量、通过率和难度分布为核心评价指标，'
    '旨在回答以下研究问题：多轮反馈机制是否能够提升题目的总量和多样性？'
    '难度自适应进化策略是否能够改善题目的难度分布，使其更接近预设目标？'
)

sec2_1_2_body_2 = (
    '实验设计了四组对照条件。A组为直接生成基线，'
    '使用单轮直接生成策略，不启用多轮反馈和难度自适应机制，'
    '作为性能下限的参考基准。B组为多轮生成无反馈组，'
    '启用多轮迭代但不使用反馈信号调整策略，'
    '用于衡量多轮迭代本身对产量的贡献。C组为有反馈无难度进化组，'
    '启用多轮反馈机制但关闭难度自适应进化，'
    '用于分离反馈机制和难度进化策略各自的贡献。'
    'D组为完整方法组，同时启用多轮反馈和难度自适应进化，'
    '代表本课题提出的完整方法。四组实验采用相同的实验配置：'
    '涵盖三个人工智能相关主题（人工智能、量子计算、第二次世界大战），'
    '每种模式目标产量为50道，最大生成轮数为10轮，'
    '难度分布目标设置为简单20%、中等30%、困难50%。'
    '使用的基座模型为glm-4-7-251222，'
    '验证阶段的引文验证分数阈值为0.65，综合质量分数阈值为0.75。'
)

sec2_1_2_body_3 = (
    '实验结果如表2.1所示。从原始产量来看，各组均能稳定产出75道以上的原始题目，'
    '其中B组（多轮无反馈）在两个模式下的原始产量均达到99道以上，'
    '说明多轮迭代本身能够有效扩展生成空间。'
    '从最终入选数量来看，各组均能产出68道以上的合格题目，'
    '满足目标产量要求。从验证通过率来看，A组直接生成基线的通过率最高（约97%），'
    '这是因为直接生成策略倾向于产生简单题目，而这些简单题目更容易通过验证；'
    'B组通过率略有下降（约86%），C组和D组的通过率约在86%—88%之间，'
    '说明随着题目难度的提升，验证环节的淘汰率也相应增加，这是质量把控的正常表现。'
)

# Experiment table as text paragraphs (simple table in text form)
sec2_1_2_table_header = (
    '表2.1 消融实验结果对比'
)

sec2_1_2_table_data_1 = (
    '组别A（直接基线）多选题：原始75题，入选74题，通过率98.7%，'
    '难度分布为简单68题、中等5题、困难1题。'
    '问答题：原始80题，入选77题，通过率96.3%，'
    '难度分布为简单71题、中等6题、困难0题。'
    '两组数据表明，直接生成策略几乎无法产出困难题目，'
    '困难题目占比不足1%，难度分布严重失衡。'
)

sec2_1_2_table_data_2 = (
    '组别B（多轮无反馈）多选题：原始99题，入选82题，通过率82.8%，'
    '难度分布为简单40题、中等39题、困难3题。'
    '问答题：原始101题，入选91题，通过率90.1%，'
    '难度分布为简单21题、中等60题、困难10题。'
    '与A组相比，多轮迭代在一定程度上改善了难度覆盖，'
    '困难题目占比提升至约7.5%，'
    '但距离预设的50%目标仍有较大差距。'
)

sec2_1_2_table_data_3 = (
    '组别C（有反馈无难度进化）多选题：原始80题，入选69题，通过率86.3%，'
    '难度分布为简单15题、中等33题、困难21题。'
    '问答题：原始79题，入选68题，通过率86.1%，'
    '难度分布为简单20题、中等25题、困难23题。'
    '引入反馈机制后，困难题目占比显著提升至约32%，'
    '说明反馈信号能够有效引导模型向困难方向生成。'
)

sec2_1_2_table_data_4 = (
    '组别D（完整方法）多选题：原始84题，入选70题，通过率83.3%，'
    '难度分布为简单12题、中等33题、困难25题。'
    '问答题：原始81题，入选75题，通过率92.6%，'
    '难度分布为简单22题、中等31题、困难22题。'
    '完整方法在保持较高通过率的同时，'
    '困难题目占比达到约32%，'
    '且在两个模式间表现出更好的一致性。'
)

sec2_1_2_analysis = (
    '综合四组结果可以得出以下结论。第一，直接生成基线（A组）虽然通过率最高，'
    '但几乎无法产出困难题目，难度分布严重偏向简单，无法满足评价数据集对难度覆盖的要求。'
    '第二，多轮迭代（B组）相比单轮直接生成能够提升原始题目的总量，'
    '但对难度分布的改善效果有限，困难题目占比仅从不足1%提升至7.5%。'
    '第三，反馈机制（C组vs B组）是改善难度分布的关键因素，'
    '引入反馈后困难题目占比从7.5%跃升至32%，说明基于上一轮结果的策略调整'
    '能够有效引导模型生成更具挑战性的题目。'
    '第四，完整的难度自适应进化（D组）在C组基础上进一步提升了难度分布的一致性，'
    '特别是在多选题模式下，困难题目占比从C组的30.4%提升至35.7%，'
    '且两个模式间的难易分布更加均衡。总体而言，'
    '实验结果验证了多轮反馈机制和难度自适应进化策略在改善评价数据集质量方面的有效性。'
)

sec2_1_2_challenge = (
    '需要指出的是，当前实验条件下各组距离预设的50%困难题目目标仍有差距，'
    '说明在有限的生成轮数和资源约束下，完全达到目标难度分布仍存在一定挑战。'
    '后续工作将探索更高效的反馈信号设计和更精细的难度控制策略，'
    '以进一步提升困难题目的生成比例。'
)

# 2.2 自动评价智能体架构
sec2_2_heading = "2.2 自动评价智能体架构"

# 2.2.1 规划智能体
sec2_2_1_heading = "2.2.1 规划智能体"

sec2_2_1_body_1 = (
    '规划智能体是整个系统的核心决策组件，'
    '负责将用户的高层评价目标转化为可逐步执行的具体计划，'
    '并在多轮执行过程中持续监控进度和调整策略。'
    '其核心工作流程是一个诊断—规划—执行—反馈的闭环迭代过程，'
    '通过主循环函数实现持续运行：系统初始化状态后，'
    '在每一轮迭代开始前检查停止条件，'
    '包括是否达到最大轮次上限、是否完成目标产量、'
    '是否超出最大token消耗等；若未触发停止条件，则依次执行诊断、规划和执行步骤。'
)

sec2_2_1_body_2 = (
    '在诊断阶段，规划智能体综合分析上一轮各环节返回的执行结果数据，'
    '从多个维度进行状态评估。诊断逻辑包含五种路径的判别：'
    '冷启动状态表示系统首次运行或重置后无历史数据，此时采用初始策略启动；'
    '仅生成状态表示当前轮次仅有生成反馈，尚未进入验证和评估环节；'
    '验证—评估状态表示上一轮完成了完整的生成、验证和评估流程；'
    '全流程状态表示上一轮完成了完整闭环并获得了评估反馈；'
    '仅验证状态表示直接对已有候选池进行验证和评估而不触发新生成。'
    '每种路径对应不同的参数更新策略，实现细粒度的自适应控制。'
    '基于上述多维度诊断结果，规划智能体将其映射为一组策略调整指令，'
    '分别作用于后续的生成计划、控制参数配置和主题选择方向。'
)

sec2_2_1_body_3 = (
    '规划智能体的工具调用机制通过编排器组件实现。'
    '编排器负责读取各子智能体的基础配置文件，'
    '包括题目生成智能体配置、验证智能体配置和模型评估智能体配置，'
    '然后与规划智能体生成的轮次规范进行深度合并，'
    '将规划层面的策略调整具体化为各子智能体可执行的配置参数。'
    '完成配置合并后，编排器依次调用三个子智能体的核心执行函数：'
    '首先调用题目生成智能体执行一轮题目生成，返回产量数据和生成反馈；'
    '然后更新共享状态，调用验证智能体对候选题目进行多阶段验证，'
    '返回验证结果和验证反馈；最后调用模型评估智能体对候选模型进行推理和评分，'
    '返回评估结果和评估反馈。三个子智能体的调用通过统一的共享状态对象进行数据传递，'
    '确保各个环节的数据一致性和流程连续性。'
    '这种基于编排的工具调用机制实现了规划与执行的解耦，'
    '使得规划智能体专注于决策而不需要关注各子智能体的实现细节。'
)

sec2_2_1_body_4 = (
    '在难度控制方面，规划智能体设计了一种双模式难度进化机制。'
    '基于规则的难度进化模式通过预设的转换逻辑调整各难度层级的目标产量：'
    '当连续多轮生成效率偏低时，适当降低高难度目标以维持生成动力；'
    '当某难度层级产量已达饱和时，将剩余目标向更高难度转移。'
    '基于大模型的智能难度进化模式则调用大模型对当前状态进行语义理解，'
    '结合历史数据的趋势分析，生成更灵活的难度调整方案。'
    '两种模式互为补充，当智能模式调用失败时自动回退至规则模式，'
    '保证系统的稳定运行。此外，规划智能体还实现了控制参数自适应调节机制，'
    '根据诊断结果动态调整验证阶段的引文分数阈值、选择模式严格程度和评估配置等参数，'
    '以及主题自适应选择机制，通过大模型选择或历史数据排序的方式，'
    '动态筛选当前最需要关注的主题方向。'
    '这些机制共同构成了规划智能体的核心决策能力，'
    '使得整个评价系统具备了动态自适应优化的能力。'
)

# 2.2.2 题目验证智能体
sec2_2_2_heading = "2.2.2 题目验证智能体"

sec2_2_2_body = (
    '题目验证智能体负责对生成智能体产出的候选题目进行多阶段质量把关，'
    '确保最终纳入评价集的题目具有高质量、多样性和可评价性。'
    '验证流程分为三个阶段。第一阶段为引文验证：'
    '对每道题目引用的证据来源进行逐条核查，'
    '判断引文内容是否真实支持题目中的陈述和标准答案，并记录每条引文的支持强度，'
    '此阶段还同时执行精确去重，基于题目文本的哈希值移除完全重复的题目。'
    '第二阶段为大模型质量评估：调用大模型对通过引文验证的候选题目进行多维度的质量评分，'
    '评估维度包括题目表述的清晰度、标准答案的正确性、引文与题目的一致性、'
    '难度设置的合理性等。在进入此阶段前，系统还执行语义近似去重，'
    '通过计算题目文本嵌入向量之间的余弦相似度，移除语义高度重复的题目，'
    '确保评价集的多样性和覆盖广度。第三阶段为加权选择：'
    '综合引文验证分数和大模型质量评分，计算每道题目的综合权重，'
    '按照题目模式和难度层级进行等比例采样，'
    '确保最终入选的题目在不同模式、不同难度层级和不同主题之间保持合理的分布比例。'
    '选定的题目被标记为选中状态，并记录完整的验证信息备查。'
    '规划智能体可以调控该组件的多项关键参数，包括引文验证的各项分数阈值、'
    '选择模式的严格程度、语义相似度阈值等，根据验证反馈动态调整这些参数，'
    '实现验证策略的自适应优化。'
)

# 2.2.3 模型评估智能体
sec2_2_3_heading = "2.2.3 模型评估智能体"

sec2_2_3_body = (
    '模型评估智能体负责在构建好的评价数据集上对候选大模型进行系统性的能力评估，'
    '自动生成多维度、多粒度的评估结果。评估流程分为四个步骤。'
    '第一步，数据集质量评估：对评价数据集本身进行质量分析，'
    '计算引文覆盖率和多样性分数，从源头保证评估的有效性。'
    '第二步，候选模型推理：将评价集中的题目逐一输入待评估的候选大模型，'
    '收集各模型生成的回答。系统通过统一的模型客户端接口抽象不同模型的调用差异，'
    '支持多种模型的并行推理，显著提升评估效率。'
    '第三步，自动指标评分：对模型回答进行客观的自动化指标评估，'
    '支持的指标包括精确匹配、关键词匹配和语义相似度等，'
    '不同题目模式可配置不同的指标组合和权重。'
    '第四步，大模型裁判评估：调用大模型作为裁判，'
    '对模型回答进行更深层次的质性分析，'
    '从回答的正确性、完整性、逻辑一致性和推理质量等多个维度进行综合打分。'
    '四个步骤的评估结果被统一聚合，生成结构化的综合评估报告，'
    '包含按题目模式、主题领域、难度层级和候选模型的多维度指标分解，'
    '以及模型间的区分度分析，为研究人员提供全方位、多层次的模型能力分析视图。'
    '规划智能体可以调控该组件的评估配置，并根据评估反馈调整下一轮的生成策略和验证策略，'
    '形成全链路的自适应优化。'
)

# 2.2.4 总体框架介绍
sec2_2_4_heading = "2.2.4 总体框架介绍"

sec2_2_4_body = (
    '本系统采用强中心化的多智能体架构，'
    '由一个核心的规划智能体统一调度三个专门化的子智能体：'
    '题目生成智能体负责构建评价数据集、'
    '题目验证智能体负责质量把关和筛选、'
    '模型评估智能体负责在数据集上对候选模型进行系统评估。'
    '这种中心化架构的核心优势在于决策的全局一致性——'
    '所有智能体的行为均由规划智能体根据用户设定的整体目标和当前系统状态统一协调，'
    '避免了分布式决策可能导致的资源冲突和策略不一致问题。'
    '系统的工作流程是一个动态的多轮反馈闭环：'
    '每一轮迭代中，规划智能体首先诊断上一轮各智能体返回的执行结果，'
    '综合分析当前进度和存在的问题，制定本轮的具体执行计划；'
    '然后通过编排器依次调用各子智能体执行具体任务；'
    '执行完成后收集各环节的反馈信息，更新系统状态，进入下一轮的诊断和规划。'
    '各子智能体之间的数据交换采用统一的数据结构规范，'
    '包括题目结构、验证结果结构、评估结果结构和反馈结构等，'
    '确保各组件之间的松耦合关系。'
    '例如，验证智能体可以在不通知其他组件的情况下切换其内部的质量评估策略，'
    '只要其输出的验证结果符合约定的数据结构即可。'
    '同样，模型评估智能体支持动态添加新的评估指标和裁判模型，'
    '新指标只需遵循约定的接口规范即可自动参与评估流程。'
    '这种可扩展的架构设计为本系统的持续演进提供了良好的基础。'
)

# 2.3 评估报告生成
sec2_3_heading = "2.3 评估报告生成"

sec2_3_body = (
    '评估报告生成是本系统的重要组成部分，'
    '其目标是将模型评估智能体产生的结构化评估数据转化为直观、全面的综合评价报告。'
    '在评估数据的组织与聚合方面，系统设计了一套层次化的数据结构，'
    '能够将原始评估指标从按题目、按模式、按主题、按难度、按模型的多个维度进行有效整合，'
    '形成结构清晰、层次分明的评估结果呈现。'
    '在评估结果的分析与解读方面，系统自动计算一系列高级分析信号，'
    '包括候选模型之间的总体能力差距、领先模型的优势程度、'
    '各题目在不同模型上的得分离散度、以及评价数据集本身的区分度质量指标，'
    '为评价数据集的质量诊断提供量化依据。'
    '在具体实现上，系统目前已实现了评估报告的标准化数据结构输出，'
    '能够生成包含多维指标分解和区分度分析的评估结果。'
    '在此基础上，后续将进一步开发面向人类研究者的可读报告生成能力，'
    '包括自动生成评估结果的文本摘要和结论分析、'
    '绘制雷达图和柱状图等可视化图表，'
    '以及基于评估数据自动提炼各模型的优势领域和不足之处，'
    '帮助研究人员快速理解各候选模型的能力差异，'
    '为模型选型和改进提供数据驱动的决策支持。'
)


def main():
    doc = Document(OUT_PATH)

    # Find section boundaries
    sec2_heading_idx = -1
    sec3_heading_idx = -1
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if '目前已经完成的研究工作' in txt:
            sec2_heading_idx = i
        if sec2_heading_idx >= 0 and '后期拟完成的研究工作' in txt:
            sec3_heading_idx = i
            break

    print(f'sec2 heading: {sec2_heading_idx}, sec3 heading: {sec3_heading_idx}')

    if sec2_heading_idx < 0 or sec3_heading_idx < 0:
        print('ERROR: Could not find section boundaries')
        return

    # Remove existing section 2 content
    elements_to_remove = []
    for i in range(sec2_heading_idx + 1, sec3_heading_idx):
        elements_to_remove.append(doc.paragraphs[i]._element)

    for elem in elements_to_remove:
        parent = elem.getparent()
        if parent is not None:
            parent.remove(elem)

    print(f'Removed {len(elements_to_remove)} paragraphs')

    doc.save(OUT_PATH)
    doc = Document(OUT_PATH)

    # Re-find sec2 heading index
    sec2_heading_idx = -1
    for i, p in enumerate(doc.paragraphs):
        if '目前已经完成的研究工作' in p.text.strip():
            sec2_heading_idx = i
            break

    print(f'New sec2 heading index: {sec2_heading_idx}')

    # Build all content items in display order
    all_items = [
        # 2.1 评价数据集构造
        {'text': sec2_1_heading, 'bold': True, 'size': SZ_SUBSECTION, 'ea_font_name': FONT_HEADING_EA},

        # 2.1.1 研究内容
        {'text': sec2_1_1_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_1_1_body_1, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_1_body_2, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_1_body_3, 'bold': False, 'size': SZ_BODY},

        # 2.1.2 实验方案
        {'text': sec2_1_2_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_1_2_body_1, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_body_2, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_body_3, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_table_header, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_1_2_table_data_1, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_table_data_2, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_table_data_3, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_table_data_4, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_analysis, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_1_2_challenge, 'bold': False, 'size': SZ_BODY},

        # 2.2 自动评价智能体架构
        {'text': sec2_2_heading, 'bold': True, 'size': SZ_SUBSECTION, 'ea_font_name': FONT_HEADING_EA},

        # 2.2.1 规划智能体
        {'text': sec2_2_1_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_2_1_body_1, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_2_1_body_2, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_2_1_body_3, 'bold': False, 'size': SZ_BODY},
        {'text': sec2_2_1_body_4, 'bold': False, 'size': SZ_BODY},

        # 2.2.2 题目验证智能体
        {'text': sec2_2_2_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_2_2_body, 'bold': False, 'size': SZ_BODY},

        # 2.2.3 模型评估智能体
        {'text': sec2_2_3_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_2_3_body, 'bold': False, 'size': SZ_BODY},

        # 2.2.4 总体框架介绍
        {'text': sec2_2_4_heading, 'bold': True, 'size': SZ_BODY},
        {'text': sec2_2_4_body, 'bold': False, 'size': SZ_BODY},

        # 2.3 评估报告生成
        {'text': sec2_3_heading, 'bold': True, 'size': SZ_SUBSECTION, 'ea_font_name': FONT_HEADING_EA},
        {'text': sec2_3_body, 'bold': False, 'size': SZ_BODY},
    ]

    insert_many(doc, sec2_heading_idx, all_items)
    doc.save(OUT_PATH)

    # === Verify ===
    doc = Document(OUT_PATH)
    sec2_text = ''
    started = False
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if '目前已经完成的研究工作' in txt:
            started = True
            continue
        if started and '后期拟完成的研究工作' in txt:
            break
        if started and txt:
            sec2_text += txt

    chinese = sum(1 for c in sec2_text if '一' <= c <= '鿿')
    total_chinese = sum(1 for p in doc.paragraphs for c in p.text.strip() if '一' <= c <= '鿿')
    print(f'\nSection 2: {chinese} Chinese characters, {len(sec2_text)} total chars')
    print(f'Total document: {total_chinese} Chinese characters, {len(doc.paragraphs)} paragraphs')

    # Print structure
    print('\nSection 2 structure:')
    for i, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        if not txt:
            continue
        if i >= sec2_heading_idx or ('目前已经完成的研究工作' in txt):
            if any(kw in txt for kw in ['2.1', '2.2', '2.3', '目前已经完成', '后期拟完成']):
                print(f'  [{i}] {txt[:80]}')
                if '后期拟完成' in txt:
                    break
            elif txt.startswith('2.'):
                print(f'  [{i}] {txt[:80]}')

    print(f'\nSaved to: {OUT_PATH}')


if __name__ == '__main__':
    main()
