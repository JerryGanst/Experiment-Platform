import requests
import os
from urllib.parse import urlparse, parse_qs
import urllib3
from tqdm import tqdm
import time

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ▶ ① 把 Meta 邮件里的“带签名”链接贴到这里
base_url = 'https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZzYwYW00YmRwYm8zMWJzeTlka2F1YmJwIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc1MDU3OTE0MX19fV19&Signature=YCv9VbZlDWsLF29ERRSmwNQCEC11rBtsIfqE2GCCotA%7E6D32fnagkwIut3t1zWuRaCuLviljD3yLzb1CgSCejZJe%7ElFfsTHprnd3bPUxPQxMXrX4l39sEZQX1C7KA47ucpuSbiw6qRV1Nt5F8TyURlt1AdzBQNlySEd0HN3K6sEDNSOGqgPtROZYUCb2PoP6B3DTHrXkDj%7EJyjn%7EfI7PBvPsgy683WZFv8J1F7ANhpErDiK7OnlxV0avpXJuTLeI%7EojDqRzcvQTHSIIQN3f83xj0nU3N%7EvsnKdVGeJ9s-9weUrvsxh5ZmZStb6i-0kEMBYF0mCmE9PbSnoIqBa43kA__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1275803444059541'

# ▶ ② Llama-2-7B-Chat 所需文件
files_to_download = [
    "consolidated.00.pth",
    "consolidated.01.pth",
    "params.json",
    "tokenizer.model",
    "checklist.chk",
]
# ▶ ③ 下载到哪里
target_dir = r"C:\Users\Administrator\llama_models\Llama-2-7b-chat"

# 统一配置
CHUNK_SIZE  = 64 * 1024       # 64 KB
TIMEOUT     = 300             # 5 min
MAX_RETRIES = 5
RETRY_DELAY = 5               # 秒


def get_file_size(url):
    try:
        print(f"  🔍 检查URL: {url}")
        r = requests.head(url, verify=False, timeout=30)
        print(f"  📊 状态码: {r.status_code}")
        if r.status_code == 200:
            size = int(r.headers.get("content-length", 0))
            print(f"  📏 文件大小: {size:,} bytes")
            return size
        else:
            print(f"  ❌ HTTP错误: {r.status_code}")
            return 0
    except Exception as e:
        print(f"  ❌ 请求错误: {e}")
        return 0


def download_file_with_resume(url, filepath, max_retries=MAX_RETRIES):
    resume_pos = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    total_size = get_file_size(url)
    if total_size == 0:
        print(f"❌ 无法获取文件大小: {os.path.basename(filepath)}")
        return False
    if resume_pos >= total_size:
        print(f"✅ 已完整: {os.path.basename(filepath)}")
        return True

    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    retry = 0
    while retry < max_retries:
        try:
            print(f"📥 {os.path.basename(filepath)}  (尝试 {retry+1}/{max_retries})")
            sess = requests.Session()
            sess.mount(
                "https://",
                requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=3),
            )
            resp = sess.get(url, headers=headers, stream=True, timeout=(30, TIMEOUT), verify=False)
            resp.raise_for_status()

            # 进度条
            pbar = tqdm(
                total=total_size,
                initial=resume_pos,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(filepath),
                ascii=True,
            )
            mode = "ab" if resume_pos else "wb"
            with open(filepath, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()

            if os.path.getsize(filepath) == total_size:
                print(f"✅ 完成: {os.path.basename(filepath)}")
                return True
            else:
                print("⚠️  文件尺寸不符，重试…")
        except Exception as e:
            print(f"❌ 错误: {e}")
        retry += 1
        print(f"⏳ {RETRY_DELAY} 秒后重试…")
        time.sleep(RETRY_DELAY)
        resume_pos = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        headers["Range"] = f"bytes={resume_pos}-"
    return False


def main():
    print("🚀 开始下载 Llama-2-7B-Chat（断点续传）")
    print(f"📁 目录: {target_dir}")
    print(f"⚙️  块={CHUNK_SIZE//1024}KB, 超时={TIMEOUT}s, 重试={MAX_RETRIES}")

    os.makedirs(target_dir, exist_ok=True)

    # ▶ ④ 仅改这里即可换模型
    MODEL_DIR = "llama-2-7b-chat"  # 使用测试确认的正确格式
    possible_formats = [
        f"/{MODEL_DIR}/{{}}?",  # 这是测试确认有效的格式！
    ]

    success = 0
    for i, fname in enumerate(files_to_download, 1):
        print(f"\n📋 文件 {i}/{len(files_to_download)}: {fname}")
        path = os.path.join(target_dir, fname)
        ok = False
        for fmt in possible_formats:
            url = base_url.replace("/*?", fmt.format(fname))
            print(f"🔗 试 URL: {fmt}")
            if download_file_with_resume(url, path):
                ok = True
                success += 1
                break
            else:
                print(f"❌ URL 失败: {fmt}")
        if not ok:
            print(f"💥 全部 URL 失败: {fname}")
        print("-" * 80)

    print(f"\n📊 成功 {success}/{len(files_to_download)}")
    if success == len(files_to_download):
        print("🎉 全部下载完成！")


if __name__ == "__main__":
    main()
