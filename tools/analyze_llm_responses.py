"""分析 LLM response 文本结构：格式、截断、多余内容"""
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

base = 'D:/Download/vscode/code/master/benchforge/runs/exp_e2e/run_20260614_221616'

with open(f'{base}/llm_calls.jsonl', encoding='utf-8') as f:
    calls = [json.loads(line) for line in f if line.strip()]

unique = [c for c in calls if c.get('llm_call_id')]

for i, c in enumerate(unique):
    req = c.get('request', {})
    resp = c.get('response', {})

    llm_id = c['llm_call_id'][:35]
    text = resp.get('text', '')
    finish = resp.get('finish_reason', '?')
    out_tok = resp.get('output_tokens', 0)
    max_tok = req.get('max_tokens', 4096)

    text_stripped = text.strip()

    # ── 1. 结构检查 ──
    arr_start = text_stripped.find('[')
    arr_end = text_stripped.rfind(']')
    before_json = text_stripped[:arr_start].strip() if arr_start > 0 else ''
    after_json = text_stripped[arr_end+1:].strip() if arr_end < len(text_stripped) - 1 else ''

    # ── 2. Prompt 中的 difficulty ──
    prompt_difficulty = "?"
    for msg in req.get('messages', []):
        content = msg.get('content', '')
        if msg.get('role') == 'system':
            # 查找 difficulty 值
            m = re.search(r'"target_difficulty"\s*:\s*"(\w+)"', content)
            if m:
                prompt_difficulty = m.group(1)
            # 也查 Target Difficulty
            m2 = re.search(r'[Tt]arget\s+[Dd]ifficulty[:\s]+(\w+)', content)
            if m2:
                prompt_difficulty = m2.group(1)
            break

    # ── 3. JSON 解析测试 ──
    json_array_text = text_stripped[arr_start:arr_end+1] if arr_start >= 0 and arr_end > arr_start else text_stripped
    parse_ok = False
    question_count = 0
    try:
        parsed = json.loads(json_array_text)
        if isinstance(parsed, list):
            parse_ok = True
            question_count = len(parsed)
        elif isinstance(parsed, dict):
            parse_ok = True
            question_count = 1
    except json.JSONDecodeError as e:
        pass

    # ── 4. 输出 ──
    print(f'=== 调用[{i}] {llm_id} ===')
    print(f'  Prompt目标难度: {prompt_difficulty}')
    print(f'  max_tokens={max_tok}, output_tokens={out_tok}, finish={finish}')
    print(f'  文本总长: {len(text_stripped)} chars')
    print(f'  首字符: [{text_stripped[0]}] 末字符: [{text_stripped[-1]}]')
    print(f'  JSON解析: {"✅ 成功" if parse_ok else "❌ 失败"}, 题目数={question_count}')

    if before_json:
        print(f'  ⚠️  JSON前多余文本 ({len(before_json)} chars):')
        print(f'      {before_json[:200]}')

    if after_json:
        print(f'  ⚠️  JSON后多余文本 ({len(after_json)} chars):')
        print(f'      {after_json[:200]}')

    # 检查截断
    if finish != 'stop':
        print(f'  ❌ 异常截断! finish={finish}')
    elif out_tok >= max_tok - 5:
        print(f'  ⚠️  可能达到max_tokens上限截断 (out={out_tok}, max={max_tok})')
    else:
        print(f'  ✅ 完整输出 (finish=stop, {out_tok}/{max_tok} tokens)')

    # 显示文本的尾部200字符
    print(f'  文本尾部: ...{text_stripped[-200:]}')
    print()
