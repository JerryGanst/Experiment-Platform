# 创建测试脚本 test_url.py
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

base_url = "你的完整URL"

# 尝试不同的路径格式
test_paths = [
    "Llama-3.1-8B-Instruct/params.json",
    "llama-3.1-8b-instruct/params.json",
    "models/Llama-3.1-8B-Instruct/params.json",
    "params.json"
]

for path in test_paths:
    test_url = base_url.replace('/*?', f'/{path}?')
    try:
        response = requests.head(test_url, verify=False, timeout=10)
        if response.status_code == 200:
            print(f"✅ 找到正确路径: {path}")
            break
    except:
        continue