
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
QUERY = os.environ.get('QUERY', 'cs.IR')
LIMITS = int(os.environ.get('LIMITS',2))
FEISHU_URL = os.environ.get("FEISHU_URL", None)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "DeepSeek")
ECNU_MODEL_KEY = os.environ.get("ECNU_MODEL_KEY", None)


from openai import OpenAI
import json
from tqdm.notebook import tqdm

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
        你是一位精通推荐系统领域的中英双语研究员，熟悉召回、排序、重排、生成式推荐等核心技术。请将以下英文论文内容（可能包含标题、摘要、方法或实验部分）**准确、流畅地翻译为中文**，并严格遵循以下原则：

        1. **术语规范**（必须遵守）：
        - recommendation / recommender system → 推荐系统  
        - user/item → 用户/物料（保留斜杠格式）  
        - recall → 召回  
        - pre-ranking → 粗排  
        - ranking → 精排（或“排序”，根据上下文）  
        - re-ranking → 重排  
        - collaborative filtering → 协同过滤  
        - embedding → 嵌入  
        - negative sampling → 负采样  
        - click-through rate (CTR) → 点击率  
        - sequential recommendation → 序列推荐  
        - graph neural network (GNN) → 图神经网络  
        - large language model (LLM) → 大语言模型  
        - retrieval → 检索（在召回上下文中可译为“召回”）  
        - exposure bias → 曝光偏差  
        - debiasing → 去偏  
        - cold start → 冷启动  
        - end-to-end → 端到端

        2. **翻译要求**：
        - 保持原文技术细节、逻辑结构和学术严谨性，**不增不减、不意译**；
        - 使用**正式学术中文**，避免口语化表达；
        - 模型名称（如 DIN、SASRec、LightGCN）、数据集名（如 MovieLens、Amazon-Book）、变量符号（如 (u), (v), (r_{{ui}})）**保留英文不译**；
        - 被动语态可适当转为主动语态以符合中文习惯，但**不得改变原意**。

        3. **输出格式**：
        - 直接输出**纯中文翻译结果**，不要包含“翻译如下：”等引导语；
        - 保留原文段落结构；
        - 不添加任何解释、注释或额外内容。

        现在，请翻译以下内容：{texts[i]}
        """
        response = client.chat.completions.create(
            model="ecnu-plus",
            messages=[
                {"role": "system", "content": "你是一位精通人工智能、推荐系统、计算机视觉和自然语言处理领域的专业学术翻译专家。请严格按照用户的要求进行翻译。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=5000

        )
        res.append(response.choices[0].message.content)
    return res

def label_and_score_paper(title, abstract):
    prompt = f"""
        你是一位资深推荐系统研究员，专注于信息检索与推荐算法领域。请根据我提供的论文**标题（Title）**和**摘要（Abstract）**，完成以下任务：

        1. **判断相关性**：  
        - 如果论文**与推荐系统/推荐算法无直接关联**（例如仅涉及通用 NLP、计算机视觉、广告投放、纯搜索排序、数据库索引等），则相关性分数为 **1–3 分**，并标注“不相关”。
        - 如果论文**明确涉及推荐系统中的召回、排序、重排、生成式推荐等环节**，则分数为 **4–10 分**，分数越高表示与推荐系统核心问题越紧密。

        2. **打分（1–10 分）**：  
        - 10 分：论文核心贡献直接解决推荐系统关键问题（如新召回架构、精排模型、LLM for Recommendation 等）。
        - 7–9 分：论文方法可直接用于推荐系统某环节，但非专为推荐设计（如通用序列建模用于推荐）。
        - 4–6 分：论文与推荐有间接联系（如用户行为建模、负采样策略），但需适配。
        - 1–3 分：与推荐系统基本无关。

        3. **打标签**：  
        从以下标签中选择**一个或多个**最贴切的（可多选，标签使用中文）：
        - 召回（Recall）
        - 粗排（Pre-ranking）
        - 精排（Ranking）
        - 重排（Re-ranking）
        - LLM生成式推荐（LLM-based Generative Recommendation）
        - 通用推荐技术（General Recommendation Techniques）
        - 多模态推荐（Multimodal Recommendation）
        - 序列推荐（Sequential Recommendation）
        - 图神经网络推荐（GNN for Recommendation）
        - 推荐系统公平性/可解释性（Fairness/Explainability）
        - 负采样与对比学习（Negative Sampling / Contrastive Learning）
        - 跨域/联邦推荐（Cross-domain / Federated Recommendation）
        - 推荐系统评估（Evaluation Metrics / Offline/Online Testing）

        > 如果论文不属于推荐领域，标签留空或写“无”。

        4. **输出格式**（严格按以下 JSON 格式，不要解释）：
        {{
        "relevance_score": 整数（1-10）,
        "tags": ["标签1", "标签2", ...],
        "reason": "简要说明打分和标签理由（30字以内）"
        }}

        现在，请处理以下论文：

        标题：{title}
        摘要：{abstract}
    """

    response = client.chat.completions.create(
        model="ecnu-plus",
        messages=[
            {"role": "system", "content": "你是一位资深推荐系统研究员，专注于信息检索与推荐算法领域。请根据用户的要求进行判断和打分。"},
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
    if 'relevance_score' not in result_json or 'tags' not in result_json or 'reason' not in result_json:
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
        msg_content = f"[{msg_title}]({url})\n\n推荐算法相关性分数：{relevance_score}\n标签：{', '.join(label)}\n标签理由：{label_reason}\n\n{msg_pub_date}\n\n{msg_url}\n\n{msg_translated}\n\n{msg_summary}\n\n"

        # send_wechat_message(push_title, msg_content, SERVERCHAN_API_KEY)
        send_feishu_message(push_title, msg_content, FEISHU_URL)

        time.sleep(12)

    print('[+] 每日推送任务执行结束')

    return True


if __name__ == '__main__':
    cronjob()