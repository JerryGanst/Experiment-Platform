
import requests

# ========== 配置区 ==========
API_BASE = "http://10.180.116.2:6400/v1"
MODEL = "openai_gpt-oss-120b"


# ============================

def ask(prompt, system="你是一个耐心的学习助手"):
    """问 AI，返回回答"""
    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        }
    )
    return resp.json()["choices"][0]["message"]["content"]


# ========== 练习区（改这里试效果） ==========
if __name__ == "__main__":
    # 练习1：直接问
    print(ask("Python 的 for 循环怎么用？"))

    # 练习2：换个角色
    print(ask(
        prompt="解释一下什么是 API",
        system="你是一个给小学生讲课的老师，用比喻解释"
    ))

    # 练习3：要求格式
    print(ask("""
用以下格式回答：
- 一句话解释：
- 举个例子：
- 常见误区：

问题：什么是递归？
"""))

    # 练习4：给例子让它学（Few-shot）
    print(ask("""
请按照示例的风格回答：

问：什么是变量？
答：变量就像一个盒子，你可以往里放东西，也可以换掉里面的东西。

问：什么是函数？
答：
"""))