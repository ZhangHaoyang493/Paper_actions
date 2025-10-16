
'''
credit to original author: Glenn (chenluda01@outlook.com)
Author: Doragd
'''

import os
import requests
import time
import json
import datetime
from tqdm import tqdm
# from translate import translate

SERVERCHAN_API_KEY = os.environ.get("SERVERCHAN_API_KEY", None)
QUERY = os.environ.get('QUERY', 'cs.CV')
LIMITS = int(os.environ.get('LIMITS',2))
FEISHU_URL = os.environ.get("FEISHU_URL", None)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "DeepSeek")
ECNU_MODEL_KEY = os.environ.get("ECNU_MODEL_KEY", None)


from openai import OpenAI
import json

client = OpenAI(
    api_key=ECNU_MODEL_KEY,
    base_url="https://chat.ecnu.edu.cn/open/api/v1"  # 注意：不包含 /chat/completions
)



def get_yesterday():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')


def search_arxiv_papers(search_term, max_results=10):
    papers = []

    url = f'http://export.arxiv.org/api/query?' + \
          f'search_query=all:{search_term}' +  \
          f'&start=0&&max_results={max_results}' + \
          f'&sortBy=submittedDate&sortOrder=descending'

    response = requests.get(url)

    if response.status_code != 200:
        return []

    feed = response.text
    entries = feed.split('<entry>')[1:]

    if not entries:
        return []

    print('[+] 开始处理每日最新论文....')

    for entry in entries:

        title = entry.split('<title>')[1].split('</title>')[0].strip()
        summary = entry.split('<summary>')[1].split('</summary>')[0].strip().replace('\n', ' ').replace('\r', '')
        url = entry.split('<id>')[1].split('</id>')[0].strip()
        pub_date = entry.split('<published>')[1].split('</published>')[0]
        pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

        papers.append({
            'title': title,
            'url': url,
            'pub_date': pub_date,
            'summary': summary,
            'translated': '',
        })
    
    print('[+] 开始翻译每日最新论文并缓存....')

    papers = save_and_translate(papers)
    
    return papers


def send_wechat_message(title, content, SERVERCHAN_API_KEY):
    url = f'https://sctapi.ftqq.com/{SERVERCHAN_API_KEY}.send'
    params = {
        'title': title,
        'desp': content,
    }
    requests.post(url, params=params)

def send_feishu_message(title, content, url=FEISHU_URL):
    card_data = {
        "config": {
            "wide_screen_mode": True
        },
        "header": {
            "template": "green",
            "title": {
            "tag": "plain_text",
            "content": title
            }
        },
        "elements": [
            {
            "tag": "img",
            "img_key": "img_v2_9781afeb-279d-4a05-8736-1dff05e19dbg",
            "alt": {
                "tag": "plain_text",
                "content": ""
            },
            "mode": "fit_horizontal",
            "preview": True
            },
            {
            "tag": "markdown",
            "content": content
            }
        ]
    }
    card = json.dumps(card_data)
    body =json.dumps({"msg_type": "interactive","card":card})
    headers = {"Content-Type":"application/json"}
    requests.post(url=url, data=body, headers=headers)


def translate(texts):
    

    res = []
    for i in range(len(texts)):
        print(f'[+] 正在翻译第 {i+1} 篇论文....')
        prompt = f"""
        你是一位精通计算机视觉，尤其是 low-level 图像处理领域的中英双语研究员。请将以下英文论文内容（可能包含标题、摘要、方法描述或实验部分）**准确、流畅地翻译为中文**，并遵循以下原则：

        1. **术语规范**（必须严格遵守）：
        - image restoration → 图像恢复  
        - image denoising → 图像去噪  
        - image deraining → 图像去雨  
        - image dehazing → 图像去雾  
        - image deblurring → 图像去模糊  
        - super-resolution → 超分辨率  
        - reflection removal → 去反射  
        - metal artifact reduction → 金属伪影消除  
        - low-light enhancement → 低光照增强  
        - JPEG artifact removal → JPEG 伪影去除  
        - degradation → 退化  
        - prior / prior knowledge → 先验 / 先验知识  
        - end-to-end → 端到端  
        - feature map → 特征图  
        - residual learning → 残差学习  
        - frequency domain → 频域  
        - spatial domain → 空域

        2. **翻译要求**：
        - 保持原文技术细节和逻辑结构，**不增不减**；
        - 使用**学术中文表达**，避免口语化；
        - 公式、变量名（如 (x), (y), (I_{{rain}})）、模型名称（如 U-Net, DnCNN）**保留原文不译**；
        - 机构名、数据集名（如 Rain100L, SIDD, RESIDE）**保留英文**；
        - 被动语态可转为主动语态以符合中文习惯，但**不得改变原意**。

        3. **输出格式**：
        - 直接输出**纯中文翻译结果**，不要包含“翻译如下：”等引导语；
        - 保留原文段落结构；

        现在，请翻译以下内容：{texts[i]}
        """
        response = client.chat.completions.create(
            model="ecnu-plus",
            messages=[
                {"role": "system", "content": "你是一位精通计算机视觉，尤其是 low-level 图像处理领域的中英双语研究员。请严格按照用户的要求进行翻译。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000

        )
        res.append(response.choices[0].message.content)
    return res

def label_and_score_paper(title, abstract):
    prompt = f"""
        你是一位专注于 low-level 计算机视觉的研究员，尤其擅长图像恢复、增强与复原任务。请根据我提供的论文**标题（Title）**和**摘要（Abstract）**，完成以下任务：

        1. **判断是否属于 low-level 图像处理**：  
        - **Low-level 任务**包括：图像恢复（如去噪、去雨、去雾、去模糊、超分辨率、去反射、去 JPEG 伪影、金属伪影消除等）、图像增强（如对比度增强、低光照增强）、图像重建等，目标是**改善或恢复图像的像素级质量**。  
        - **High-level 任务**（不相关）包括：图像分类、目标检测、语义分割（若用于场景理解）、人脸识别、图像生成（如 GAN 生成新图像，而非恢复）、视频理解、3D 重建（非图像级）等。  
        - 注意：**图像分割**若用于**医学图像分割、边缘提取、显著性检测等像素级恢复/分析任务**，可视为 low-level；若用于场景理解（如 Cityscapes 分割），则视为 high-level。

        2. **相关性打分（1–10 分）**：  
        - **10 分**：论文核心贡献直接解决 low-level 图像处理问题（如提出新去雨架构、CT 金属伪影消除方法）。  
        - **7–9 分**：方法专为 low-level 设计，或在标准 low-level 数据集（如 Rain100L、SIDD、RESIDE）上验证。  
        - **4–6 分**：技术可迁移到 low-level（如通用退化建模、频域先验），但非主要目标。  
        - **1–3 分**：属于 high-level 视觉任务，或仅使用图像作为输入但不处理像素质量。

        3. **创新度打分（1–10 分）**：  
        - **10 分**：提出全新范式、理论或架构（如首次将扩散模型用于真实图像去雨）。  
        - **7–9 分**：对现有方法有显著改进（如新损失函数、高效网络结构），性能或效率明显提升。  
        - **4–6 分**：常规组合或微调（如换 backbone、调超参），无本质创新。  
        - **1–3 分**：复现已有工作、方法陈旧，或创新点模糊不清。

        4. **打标签**：  
        从以下标签中选择**一个或多个**最贴切的（可多选，标签使用中文）：
        - 图像恢复（Image Restoration）
        - 图像去噪（Image Denoising）
        - 图像去雨（Image Deraining）
        - 图像去雾（Image Dehazing）
        - 图像去模糊（Image Deblurring）
        - 超分辨率（Super-Resolution）
        - 图像去反射（Image Reflection Removal）
        - 图像去 JPEG 伪影（JPEG Artifact Removal）
        - CT金属伪影消除（CT Metal Artifact Reduction）
        - 低光照增强（Low-light Enhancement）
        - 对比度增强（Contrast Enhancement）
        - 图像修复（Image Inpainting）
        - 多帧/视频图像恢复（Video/Image Sequence Restoration）
        - 医学图像增强（Medical Image Enhancement）
        - 遥感图像复原（Remote Sensing Image Restoration）

        > 若论文不属于 low-level 图像处理，标签留空。

        5. **输出格式**（严格按以下 JSON 格式，不要任何额外解释、注释或 Markdown）：
        {{
        "relevance_score": 整数（1-10）,
        "novelty_score": 整数（1-10）,
        "tags": ["标签1", "标签2", ...],
        "reason": "相关性与标签理由（30字以内）",
        "novelty_reason": "创新度打分理由（30字以内）"
        }}

        现在，请处理以下论文：

        标题：{title}
        摘要：{abstract}
    """

    response = client.chat.completions.create(
        model="ecnu-plus",
        messages=[
            {"role": "system", "content": "你是一位专注于 low-level 计算机视觉的研究员，尤其擅长图像恢复、增强与复原任务。请严格按照用户的要求进行打标和打分。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=5000
    )

    result = response.choices[0].message.content.strip()
    try:
        result_json = json.loads(result)
    except json.JSONDecodeError:
        print(f"[ERROR] JSON Decode Error: {result}")
        return None
    if not isinstance(result_json, dict):
        print(f"[ERROR] Result is not a JSON object: {result}")
        return None
    if 'relevance_score' not in result_json or 'tags' not in result_json or 'reason' not in result_json or 'novelty_score' not in result_json or 'novelty_reason' not in result_json:
        print(f"[ERROR] Missing keys in result: {result_json}")
        return None
    if not isinstance(result_json['relevance_score'], int) or not (1 <= result_json['relevance_score'] <= 10):
        print(f"[ERROR] Invalid relevance score: {result_json['relevance_score']}")
        return None
    if not isinstance(result_json['tags'], list):
        print(f"[ERROR] Tags should be a list: {result_json['tags']}")
        return None
    return result_json

def label_and_score_papers(papers):
    results = []
    for i, paper in enumerate(papers):
        print(f'[+] 正在打标和打分第 {i+1} 篇论文')
        title = paper['title']
        abstract = paper['summary']
        result = label_and_score_paper(title, abstract)
        results.append(result)
    return results


def save_and_translate(papers, filename='arxiv.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        results = json.load(f)

    cached_title2idx = {result['title'].lower():i for i, result in enumerate(results)}
    
    untranslated_papers = []
    translated_papers = []
    for paper in papers:
        title = paper['title'].lower()
        if title in cached_title2idx.keys():
            translated_papers.append(
                results[cached_title2idx[title]]
            )
        else:
            untranslated_papers.append(paper)
    
    source = []
    titles_translate = []
    for paper in untranslated_papers:
        source.append(paper['summary'])
        titles_translate.append(paper['title'])
    target = translate(source)
    titles_target = translate(titles_translate)
    label_score_results = label_and_score_papers(untranslated_papers)
    if len(target) == len(untranslated_papers) and len(titles_target) == len(untranslated_papers) and len(label_score_results) == len(untranslated_papers):
        for i in range(len(untranslated_papers)):
            untranslated_papers[i]['translated'] = target[i]
            untranslated_papers[i]['translated_title'] = titles_target[i]
            if label_score_results[i] is not None:
                untranslated_papers[i]['label'] = label_score_results[i]['tags']
                untranslated_papers[i]['label_reason'] = label_score_results[i]['reason']
                untranslated_papers[i]['relevance_score'] = label_score_results[i]['relevance_score']
                untranslated_papers[i]['novelty_score'] = label_score_results[i]['novelty_score']
                untranslated_papers[i]['novelty_reason'] = label_score_results[i]['novelty_reason']
            else:
                untranslated_papers[i]['label'] = []
                untranslated_papers[i]['label_reason'] = ''
                untranslated_papers[i]['relevance_score'] = 0
                untranslated_papers[i]['novelty_score'] = 0
                untranslated_papers[i]['novelty_reason'] = ''
    results.extend(untranslated_papers)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'[+] 总检索条数: {len(papers)} | 命中缓存: {len(translated_papers)} | 实际返回: {len(untranslated_papers)}....')

    return untranslated_papers # 只需要发送缓存中没有的

        
def cronjob():

    if SERVERCHAN_API_KEY is None:
        raise Exception("未设置SERVERCHAN_API_KEY环境变量")

    print('[+] 开始执行每日推送任务....')

    yesterday = get_yesterday()
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    print('[+] 开始检索每日最新论文....')
    papers = search_arxiv_papers(QUERY, LIMITS)

    if papers == []:
        
        push_title = f'Arxiv:{QUERY}[X]@{today}'
        send_wechat_message('', '[WARN] NO UPDATE TODAY!', SERVERCHAN_API_KEY)

        print('[+] 每日推送任务执行结束')

        return True
        

    print('[+] 开始推送每日最新论文....')

    for ii, paper in enumerate(tqdm(papers, total=len(papers), desc=f"论文推送进度")):

        title = paper['title']
        url = paper['url']
        pub_date = paper['pub_date']
        summary = paper['summary']
        translated = paper['translated']
        translated_title = paper['translated_title']
        label = paper['label']
        label_reason = paper['label_reason']
        relevance_score = paper['relevance_score']
        novelty_score = paper['novelty_score']
        novelty_reason = paper['novelty_reason']

        yesterday = get_yesterday()

        if pub_date == yesterday:
            msg_title = f'[Newest]{title}' 
        else:
            msg_title = f'{title}'

        msg_url = f'URL: {url}'
        msg_pub_date = f'Pub Date：{pub_date}'
        msg_summary = f'Summary：\n\n{summary}'
        msg_translated = f'Translated (Powered by {MODEL_TYPE}):\n\n{translated}'

        push_title = f'Arxiv: {translated_title}[{ii}]@{today}'
        msg_content = f"[{msg_title}]({url})\n\nlow-level相关性分数：{relevance_score}\n标签：{', '.join(label)}\n标签理由：{label_reason}\n\n创新度：{novelty_score}\n创新度理由：{novelty_reason}\n\n{msg_pub_date}\n\n{msg_url}\n\n{msg_translated}\n\n{msg_summary}\n\n"

        # send_wechat_message(push_title, msg_content, SERVERCHAN_API_KEY)
        send_feishu_message(push_title, msg_content, FEISHU_URL)

        time.sleep(12)

    print('[+] 每日推送任务执行结束')

    return True


if __name__ == '__main__':
    cronjob()