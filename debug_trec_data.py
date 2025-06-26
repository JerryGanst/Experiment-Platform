import json

print("=== 检查TREC数据集结构 ===")

with open('data/trec.jsonl', 'r', encoding='utf-8') as f:
    line = f.readline()
    data = json.loads(line)
    
    print("原始数据结构:")
    for key, value in data.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== 关键字段分析 ===")
    print(f"input字段: {data.get('input', 'NOT FOUND')}")
    print(f"answers字段: {data.get('answers', 'NOT FOUND')}")
    
    # 检查正确答案
    if 'answers' in data:
        answers = data['answers']
        print(f"正确答案类型: {type(answers)}")
        print(f"正确答案内容: {answers}")
    
    # 从input中提取问题
    input_text = data.get('input', '')
    if 'Question:' in input_text:
        question_part = input_text.split('Type:')[0].strip()
        print(f"问题部分: {question_part}") 