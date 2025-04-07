import pandas as pd
import requests
import json
from uuid import uuid4
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 配置智谱API
API_KEY = "c93bc6e676f64286942b2f9809ba106d.FluBnF8yIzNmHqbw"
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def classify_news_batch(batch_data):
    """批量处理新闻分类"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    results = []
    for title, text in batch_data:
        payload = {
            "model": "glm-4-plus",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的新闻分类器，请严格按规则分类"
                },
                {
                    "role": "user",
                    "content": f"根据以下内容分类（仅返回market/policy/technology/security）：标题：{title} 正文：{text[:500]}"  # 减少文本长度
                }
            ],
            "request_id": str(uuid4()),
            "do_sample": False,
            "temperature": 0.1,
            "top_p": 0.3,
            "max_tokens": 50,
            "response_format": {"type": "text"},
            "stop": ["\n\n"],
            "user_id": "news_classifier"
        }
        
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=15  # 减少超时时间
            )
            response.raise_for_status()
            result = response.json()
            results.append(result['choices'][0]['message']['content'].strip().lower())
        except Exception as e:
            print(f"处理失败: {str(e)}")
            results.append("error")
            
        time.sleep(0.1)  # 添加小延迟避免请求过快
    
    return results

def process_file(input_path, output_path, batch_size=10, max_workers=5):
    """处理输入文件并保存分类结果"""
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("仅支持CSV/JSON格式")
    
    if not all(col in df.columns for col in ['title', 'text']):
        raise KeyError("输入文件必须包含title和text列")
    
    # 准备批处理数据
    data = list(zip(df['title'], df['text']))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    
    print("开始分类处理...")
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(classify_news_batch, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理进度"):
            results.extend(future.result())
    
    df['category'] = results
    
    if output_path.endswith('.csv'):
        df.to_csv(output_path, index=False)
    else:
        df.to_json(output_path, orient='records', lines=True)
    
    print(f"处理完成！结果保存至：{output_path}")

if __name__ == "__main__":
    process_file(
        input_path="news\cryptonewsResearch.csv",
        output_path="news\classified_data.csv",
        batch_size=10,  # 每批处理10条
        max_workers=5   # 最多5个线程
    )