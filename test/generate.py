import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 搜索路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from benchforge.config import QuestionGeneratorConfig
from benchforge.models import OpenAIClient
from benchforge.schemas import GenerationPlan, QuestionModeTarget

async def run():
    # 创建配置
    config = QuestionGeneratorConfig.from_yaml("benchforge/config/question_generator_config.yaml")
    
    # 创建计划 - 生成50道题目，涵盖多个主题和目标
    plan = GenerationPlan(
        task_id="mmlu_eval_001",
        run_id="run_20260527",
        goal="comprehensive evaluation across multiple domains: generate diverse questions for assessing reasoning capabilities in history, science, and technology",
        topics=[
            # 历史与人文领域
            "Renaissance Art",
            "Industrial Revolution",
            # 科学与数学领域
            "Quantum Mechanics",
            "Machine Learning",
            "Climate Science",
            # 技术与创新领域
            "Blockchain Technology",
            "Space Exploration"
        ],
        mode_targets={
            "qa": QuestionModeTarget(
                count=30,
                difficulty_distribution={"easy": 0.25, "medium": 0.45, "hard": 0.30}
            ),
            "multiple_choice": QuestionModeTarget(
                count=20,
                difficulty_distribution={"easy": 0.30, "medium": 0.50, "hard": 0.20}
            )
        },
        max_rounds_per_topic=8,  # 每个主题最多8轮
        max_total_rounds=40,    # 总共最多40轮
        language="en",
        retrieval_policy="wikipedia_first"
    )
    
    # 创建模型客户端
    client = OpenAIClient(
        api_key="8d955a02-ee56-48e0-890c-8c38de4a4274",
        model_name="deepseek-v3-2-251201",
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )
    
    # 创建并执行 Agent
    from benchforge.agents.question_generator.plan_driven import PlanDrivenQuestionGenerationAgent
    agent = PlanDrivenQuestionGenerationAgent(config, client)
    report = await agent.execute(plan)
    
    print(f"完成状态: {report.status}")
    print(f"最终计数: {report.final_counts}")

asyncio.run(run())
