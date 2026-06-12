import json
import time
from pathlib import Path
from openai import OpenAI
INTENT_LABELS = [
    "action_on", "action_off", "navigation", "music_play", "music_pause",
    "call_answer", "call_reject", "window_open", "window_close",
    "seat_adjust", "chitchat", "greeting", "unknown"
]
client=OpenAI(base_url="http://localhost:11434/v1",api_key="'ollama")
MODEL='qwen2.5:7b'
DATA_PATH=Path(__file__).parent.parent/'data'/"testcases.jsonl"
RESULTS_PATH=Path(__file__).parent.parent/'results'
def load_testcases():
  cases=[]
  with open(DATA_PATH) as f:
    for line in f:
      cases.append(json.loads(line.strip()))
  return cases

def predict_intent(query:str):
  prompt =f"""你是一个智能座舱语音助手。
  用户说：{query}
  请根据用户说的话，来精准判断用户意图，只输出意图标签，不要解释。
  从以下标签中选择最匹配的意图，必须严格按JSON格式输出，不要输出任何其他内容：
  {json.dumps(INTENT_LABELS, ensure_ascii=False)}
  可以在以上标签中选择任意个数，只需要能识别用户的完整意图即可，最终输出包含用户完整意图的所有意图标签。
  输出格式：{{"intent": "标签名"}}
  """
  start=time.perf_counter()
  response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": "你是智能座舱语音意图识别系统。只能输出JSON格式，不能输出任何其他内容。"
        },
        {
            "role": "user", 
            "content": prompt
        }
    ],
    temperature=0,
)
  latency=time.perf_counter()-start
  raw = response.choices[0].message.content.strip()
  try:
     parsed=json.loads(raw)
     label=parsed.get("intent",'unknown')
     if label not in INTENT_LABELS:
        label='unknown'
  except json.JSONDecodeError:
     label=raw if raw in INTENT_LABELS else "unknown"
  return label,latency
     


def evaluate():
  cases=load_testcases()
  results=[]
  for case in cases:
    predicted,latency=predict_intent(case["query"])
    correct=predicted==case["expected_intent"]
    results.append({
      "query": case["query"],
      "expected": case["expected_intent"],
      "predicted": predicted,
      "correct": correct,
      "latency": round(latency, 3),
      "type": case["type"]
    })
    status='√'if correct else 'X'
    print(f"{status}[{case['type']:8}]{case['query']!r:20}->{predicted}")
    # 统计
    total = len(results)
    correct_count = sum(r["correct"] for r in results)
    by_type = {}
    for r in results:
        t = r["type"]
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["total"] += 1
        by_type[t]["correct"] += r["correct"]
    
    print(f"\n总体准确率: {correct_count}/{total} = {correct_count/total:.1%}")
    for t, stats in by_type.items():
        print(f"{t:10}: {stats['correct']}/{stats['total']} = {stats['correct']/stats['total']:.1%}")
    avg_latency = sum(r["latency"] for r in results) / total
    print(f"平均延迟: {avg_latency:.3f}s")
    
    # 保存结果
    RESULTS_PATH.mkdir(exist_ok=True)
    latest_file = RESULTS_PATH / "latest.json"

    with open(latest_file, "w", encoding="utf-8") as f:
       json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"最新结果: {latest_file}")

if __name__ == "__main__":
   evaluate()



 