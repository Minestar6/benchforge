执行计划 v2

目标

- 让 planner 的 topic 选择从“backlog 直接排序”升级为“反馈摘要 -> topic 扩展 -> 排序裁剪”的稳定工具链
- 让首轮 topics、评估指标、难度分布可以由规划智能体基于用户目标自动生成
- 让 evaluation_requirements 在首轮生成后冻结，后续轮次不得漂移
- 明确首轮探索策略，使第一轮以较低成本试探用户目标对应的 topic 和难度空间

一、先统一协议层，再改策略层

实施顺序固定为：

1. 统一 derived_performance_signals 命名和结构
2. 定义 topic 工具链职责边界
3. 收口 evaluation_requirements 冻结真源
4. 增加首轮 bootstrap 控制逻辑
5. 扩展 topic_feedback_summarizer 输入面
6. 补测试并验证

原因：

- 1 和 2 是协议层
- 3 是真源层
- 4 和 5 是策略层
- 如果顺序反过来，后续很容易返工

二、统一 topic 级失败信号命名

问题：

当前全局字段和分组字段命名不一致，容易导致 planner 读取错误。

统一规则：

- 全局字段统一叫 all_models_fail_ratio
- topic / difficulty / mode 分组字段统一叫 all_models_fail_ratio
- 不再混用 all_fail_ratio

推荐落点：

- agents/planner_agent/schema.py
  - 明确 EvaluatorFeedback.derived_performance_signals 的约定结构
- agents/model_eval_agent 产出 evaluation_report.json 的逻辑
  - 统一输出字段名为 all_models_fail_ratio
- agents/planner_agent/planner.py 和后续 topic 工具链读取层
  - 长期只认 all_models_fail_ratio

迁移要求：

- 过渡期可以兼容旧数据：先读 all_models_fail_ratio，没有再 fallback 到 all_fail_ratio
- 但 fallback 只作为迁移措施，不能作为长期协议

建议补充的结构约定：

- derived_performance_signals.global
  - overall_model_gap
  - discriminative_question_ratio
  - easy_questions_too_easy_ratio
  - all_models_fail_ratio
- derived_performance_signals.by_topic[topic]
  - question_count
  - model_gap
  - too_easy_ratio
  - all_models_fail_ratio
- derived_performance_signals.by_difficulty[level]
  - question_count
  - model_gap
  - all_models_fail_ratio
- derived_performance_signals.by_mode[mode]
  - question_count
  - model_gap
  - all_models_fail_ratio

如果当前 schema 不便一次性强类型化，至少先在 schema.py 中把字段约定写成明确注释或辅助结构，避免继续裸用 dict[str, Any] 且无协议说明。

三、定义 topic 工具链职责边界

在 agents/planner_agent 新增：

- topic_feedback_summarizer()
- topic_expander()
- topic_ranker()

建议落点：

- 新文件：agents/planner_agent/topic_search.py

职责划分：

1. topic_feedback_summarizer

输入：

- PlannerState
- GeneratorFeedback | None
- ValidatorFeedback | None
- EvaluatorFeedback | None

输出至少包含：

- strong_topics
- weak_topics
- noisy_topics
- oversaturated_topics
- low_yield_topics
- under_observed_topics
- topic_stats

分类语义建议：

- strong_topics
  - 区分度高，验证质量稳定，可继续细分扩展
- weak_topics
  - 区分度弱，或连续多轮效果差，应替换为相邻但更稳或更专门 topic
- noisy_topics
  - 重复高、质量不稳、验证波动大
- oversaturated_topics
  - too_easy_ratio 高，说明已被过度开发，需要向更复杂、更组合式 topic 扩展
- low_yield_topics
  - 生成阶段就产能低、证据差或候选不足
- under_observed_topics
  - 样本太少，不能做强判断，只允许轻度探索或保守保留

2. topic_expander

输入：

- user_goal
- 当前 active / deferred / historical topics
- summarizer 输出的 topic 分类
- 可选的最近几轮历史统计

职责：

- 只负责调用 LLM 生成候选扩展 topic 和简短理由
- 不负责最终排序
- 不负责预算裁剪
- 不负责历史惩罚

扩展规则：

- 强 topic：扩展到相邻细分 topic
- 弱 topic：替换为相邻但更专门或更稳的 topic
- all_models_fail_ratio 高的 topic：回退到上位或邻近 topic
- too_easy_ratio 高的 topic：扩到更复杂、更组合式的 topic
- under_observed topic：只做轻量补样，不做激进扩张

LLM 只负责：

- 生成候选扩展 topic
- 给出简短理由

3. topic_ranker

输入建议：

- candidate_topics
- budget
- history
- summarized_feedback

职责边界：

- 去重
- 历史使用惩罚
- 结合 strong / weak / low_yield / under_observed 等信号排序
- 在给定预算下输出最终 selected_topics

明确禁止：

- planner.py 不再对 selected_topics 做第二次截断
- _build_next_round_spec() 不再根据 proposed_topics 长度反推 topics_per_round

单一真源规则：

- adjust_next_round_plan() 只负责决定 topics_per_round 这个预算
- topic_ranker() 只负责在预算下选出最终 topic 列表
- planner 主流程只消费 selected_topics 和 topics_per_round，不再自己裁剪

推荐接口：

- adjust_next_round_plan(...) -> NextRoundControlPlan(topics_per_round=N)
- topic_ranker(candidate_topics, budget=N, history=..., feedback=...) -> selected_topics

四、首轮 topics、评估指标、难度由规划智能体自动生成

首轮允许规划智能体基于 user_goal 自动生成：

- seed_topics
- evaluation_requirements
- mode 默认难度分布
- 首轮探索使用的建议难度倾向

这里的“自动生成”指：

- 由 blueprint_synthesizer 或拆出的首轮设计工具统一生成
- 但必须受现有 mode 和 metric 支持范围约束
- 自动生成不等于后续轮次继续漂移

建议拆分当前 blueprint_synthesizer.py 中职责，增加独立函数：

- design_evaluation_requirements(intent, model_client, registry_path)
- design_initial_mode_defaults(intent, model_client)
- 或者统一为一个首轮设计函数，但内部职责分开

规则要求：

- qa 需要由规划智能体生成 llm_judge_metrics
- multiple_choice 不参与大模型评估
- dataset_metrics 不作为 planner 的动态控制项
- 生成结果必须经过 normalize，并限制在当前系统支持的 mode / metric 范围内

注意：

- 这不要求去掉首轮 LLM 对 seed_topics/default_modes 的参与
- 真正要限制的是后续轮次不能重新定义这些评估指标

五、增加 evaluation_requirements 冻结机制

冻结原则：

- 首轮生成 GlobalBlueprint 后，把完整 evaluation_requirements 视为冻结真源
- 后续任何轮次都只能引用它，不能重新拼装 metric definition
- 冻结机制不是“加锁标记”本身，而是后续代码路径都不再拥有改写 metric definition 的权力

推荐真源：

- 优先直接使用 GlobalBlueprint.evaluation_requirements 作为唯一真源
- 如确有需要，也可以在 PlannerState 中冗余保存 frozen_evaluation_requirements 作为快照，但不应出现两套逻辑真源

后续轮次只允许调整：

- eval_profile
- judge.enabled
- candidate_model_names
- judge_model_name

后续轮次禁止调整：

- automatic_metrics
- llm_judge_metrics
- metric descriptions
- 任何 name -> description=name 的降级重写

必须收口的下游读路径：

1. agents/planner_agent/utils.py
   - translate_eval_profile() 只能做“评估强度翻译”
   - 不能改 metric 定义
2. agents/planner_agent/planner.py
   - _build_next_round_spec() 不要再内联构造 llm_judge_metrics
   - 如果 model_eval_agent_patch 需要 metrics，应从冻结真源拷贝原始定义，而不是重新生成

额外要求：

- multiple_choice 不参与大模型评估必须落实到协议层，而不是只停留在 prompt 约定
- 即：首轮设计、eval_profile 翻译、round patch 生成三处都要保证 MC 不被重新注入 judge metrics

六、增加首轮 bootstrap 控制逻辑

问题：

当前 cold_start 更像“无反馈时的默认回退”，不等于“首轮探索策略”。
第一轮应该有单独的 bootstrap 规则，而不是直接复用后续轮诊断路径。

建议：

- 新增首轮专用控制逻辑，例如：
  - build_initial_round_control_plan()
- 不要把首轮和 diagnose_last_round() 共用为同一入口

首轮控制目标：

- 以较低成本探索 topic 空间
- 对规划智能体分析出的难度倾向做小规模试探
- 让第一轮结果为后续自适应 topic 扩展和难度调整提供可靠反馈

首轮应自动生成：

- 首轮 topics 数量
- 首轮每个 mode 的题量
- 首轮难度分布

数量分配原则：

- 首轮题目数量不是最终目标数，而是探索预算
- 探索预算应基于总目标数乘以一个首轮探索比例
- 不同 mode 再根据总目标数和智能体分析出的难度比例进行分配

必须先统一“数量”的语义：

- final_targets 表示总目标数
- round_spec.blueprint.modes[*].count 表示本轮生成目标数
- validation / evaluation 后的 selected 才是完成进度

建议不要混淆：

- 生成候选数
- 本轮目标题数
- 最终入选题数

首轮探索比例建议：

- 每个 mode 的首轮目标数 = max(min_per_mode, min(max_per_mode, round(total_target * bootstrap_ratio)))
- bootstrap_ratio 建议范围：10% 到 25%
- 需要设置 min / max 下限，避免：
  - 总目标数很小时按比例变成 0
  - 总目标数很大时首轮探索过重

建议默认策略：

- QA 和 MC 分别按各自 total_target 计算 bootstrap count
- 如果某 mode total_target 很小，至少保留 1 个探索样本
- 首轮 topic 数量少于后续轮，避免成本过高和探索面过散

难度分配原则：

- 首轮难度比例来自“总目标数 + LLM 对 user_goal 难度倾向的分析”
- 首轮不是简单照抄最终默认难度比例，而是偏向探索性分布
- 如果 user_goal 明显偏研究/高难任务，可适度提高 hard 比例
- 但首轮仍应保留一定 medium 样本，避免纯 hard 导致全失败

七、扩大 topic_feedback_summarizer 的输入面

不能只吃 ValidatorFeedback.by_topic 和 EvaluatorFeedback.derived_performance_signals["by_topic"]。
应改为三层输入：

1. GeneratorFeedback
   - 用于识别：
     - 产能不足 topic
     - 证据不足 topic
     - 本轮根本没成功生成的 topic
2. ValidatorFeedback
   - 用于识别：
     - 质量差 topic
     - 重复高 topic
     - citation / llm validation 不稳定 topic
3. EvaluatorFeedback
   - 用于识别：
     - 区分度强/弱 topic
     - 过易 topic
     - 全失败 topic

建议同时引入历史视角：

- 从 PlannerState.run_history 或单独 topic 历史统计看某 topic 是连续弱，还是本轮偶然弱
- 避免仅凭单轮 by_topic 做过强决策

八、建议修改点清单

1. schema

- agents/planner_agent/schema.py
  - 明确 derived_performance_signals 协议结构
  - 如需要，为冻结快照增加字段
  - 明确 PlannerState / RoundSpec 中“题量”的语义

2. 首轮设计

- agents/planner_agent/blueprint_synthesizer.py
  - 拆出评估指标设计逻辑
  - 拆出首轮 mode defaults / difficulty 设计逻辑
  - 约束 qa / multiple_choice 的评估参与方式

3. planner 主循环

- agents/planner_agent/planner.py
  - 首轮使用 build_initial_round_control_plan()
  - 后续轮次继续走 diagnose_last_round() + adjust_next_round_plan()
  - _propose_topics_with_llm() 重构为 topic 工具链入口
  - _build_next_round_spec() 不再二次裁 topic
  - _build_next_round_spec() 不再内联重写 llm_judge_metrics

4. 评估 profile 翻译

- agents/planner_agent/utils.py
  - translate_eval_profile() 只做 profile -> patch 翻译
  - metrics 只从冻结真源读取，不重新生成

5. model_eval_agent 产物协议

- agents/model_eval_agent 相关模块
  - 统一 evaluation_report.json 中的 all_models_fail_ratio 命名
  - 保证 MC 不参与大模型 judge 路径

九、测试计划

先写失败测试，再实现。

建议新增测试放在：

- test/test_planner_agent_units.py

新增测试建议：

- test_topic_feedback_summarizer_classifies_topics_from_generator_validation_and_eval_feedback
- test_topic_feedback_summarizer_marks_under_observed_topics_when_sample_size_is_low
- test_topic_ranker_prefers_strong_topics_and_penalizes_reused_topics
- test_topic_ranker_respects_budget_as_single_source_of_truth
- test_topic_expander_llm_output_is_filtered_against_history
- test_initial_round_control_plan_uses_bootstrap_ratio_and_goal_difficulty
- test_initial_round_control_plan_preserves_nonzero_counts_for_small_targets
- test_evaluation_requirements_are_frozen_after_first_round
- test_subsequent_rounds_can_change_eval_profile_but_not_metric_definitions
- test_multiple_choice_does_not_receive_llm_judge_metrics_in_later_rounds
- test_planner_prefers_all_models_fail_ratio_and_only_falls_back_during_migration

如果 schema 调整较多，可补：

- test/test_planner_e2e.py
- test/test_model_eval_agent_units.py
- test/test_benchforge_entrypoint.py

十、验证

至少运行：

- pytest test/test_planner_agent_units.py test/test_planner_e2e.py test/test_model_eval_agent_units.py -q

如果需要更稳，再补：

- pytest test/test_benchforge_entrypoint.py -q

注意：

- 如果 entrypoint 测试当前存在与本次改动无关的旧失败，先区分是否是 prompt 路径或既有夹具问题，不要把无关问题混入本次改动

十一、交付标准

- planner 的 topic 选择不再只是 backlog 排序，而是走“反馈摘要 -> 主题扩展 -> 排序裁剪”
- derived_performance_signals 中全局和分组失败信号统一使用 all_models_fail_ratio
- 首轮 topics、评估指标、难度分布可以由规划智能体基于 user_goal 自动生成
- 首轮有独立 bootstrap 逻辑，而不是复用普通 cold_start 逻辑
- 首轮题量按总目标数和 bootstrap_ratio 控制，且有合理的 min / max 边界
- qa 会生成 llm_judge_metrics，multiple_choice 不参与大模型评估
- evaluation_requirements 首轮后不可漂移
- 后续轮次只允许调整 eval_profile、judge.enabled、candidate_model_names、judge_model_name
- planner 主控制仍由规则工具驱动，LLM 只参与首轮设计和 topic 扩展
- 全部新增行为有单测覆盖
