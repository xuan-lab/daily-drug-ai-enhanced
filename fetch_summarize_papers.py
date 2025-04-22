# scripts/fetch_summarize_papers.py
import requests
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sys # Import sys for exit

# --- 配置 ---
PUBMED_SEARCH_TERM = "pharmacy OR pharmaceutical sciences" # PubMed 搜索关键词
MAX_PAPERS_TO_FETCH = 10  # 获取和总结的最大论文数量 (调整为更合理的数量)
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions" # DeepSeek API 地址 (确认是 /chat/completions)
DEEPSEEK_MODEL = "deepseek-chat" # 使用的模型 (请确认)
OUTPUT_DIR = "output" # 输出目录
DAYS_TO_SEARCH = 1 # 搜索过去多少天的论文

# --- PubMed API 函数 ---
def search_pubmed(query, days=DAYS_TO_SEARCH, retmax=MAX_PAPERS_TO_FETCH * 2): # Use constants
    """搜索 PubMed 获取最近指定天数内的论文 ID"""
    print(f"Searching PubMed for '{query}' in the last {days} day(s)...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"

    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    # PubMed 日期格式 YYYY/MM/DD
    date_query = f"(\"{start_date.strftime('%Y/%m/%d')}\"[Date - Publication] : \"{end_date.strftime('%Y/%m/%d')}\"[Date - Publication])"

    params = {
        "db": "pubmed",
        "term": f"({query}) AND {date_query}",
        "retmode": "json",
        "retmax": retmax, # 获取稍多一些以防没摘要或过滤掉
        "sort": "pub+date" # 按发布日期排序 (最新的在前)
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status() # 检查请求是否成功
        data = response.json()
        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            id_list = data['esearchresult']['idlist']
            print(f"Found {len(id_list)} paper IDs.")
            return id_list
        else:
            print("No paper IDs found or error in response format.")
            print("Response:", data)
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error searching PubMed: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during PubMed search: {e}")
        return []


def fetch_pubmed_abstracts(ids):
    """根据 PubMed ID 获取论文摘要和标题"""
    if not ids:
        return {}
    print(f"Fetching abstracts for {len(ids)} paper IDs...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    fetch_url = f"{base_url}efetch.fcgi"
    ids_str = ",".join(ids)
    params = {
        "db": "pubmed",
        "id": ids_str,
        "retmode": "xml",
        "rettype": "abstract"
    }

    papers = {}
    try:
        response = requests.get(fetch_url, params=params)
        response.raise_for_status()

        # 解析 XML
        root = ET.fromstring(response.content)
        for article in root.findall('.//PubmedArticle'):
            pmid_element = article.find('.//PMID')
            pmid = pmid_element.text if pmid_element is not None else None
            if not pmid: continue

            title_element = article.find('.//ArticleTitle')
            # 处理可能的 XML 结构嵌套，获取完整标题文本
            title = "".join(title_element.itertext()).strip() if title_element is not None else "No Title Available"

            abstract_text = ""
            abstract_elements = article.findall('.//Abstract/AbstractText')
            if abstract_elements:
                # 处理分段摘要，并保留段落结构
                abstract_text = "\n\n".join(
                    "".join(part.itertext()).strip() for part in abstract_elements if "".join(part.itertext()).strip()
                )

            if abstract_text: # 只处理有摘要的论文
                papers[pmid] = {"title": title, "abstract": abstract_text}
                print(f"Fetched abstract for PMID: {pmid}")
            else:
                print(f"Skipping PMID: {pmid} (No abstract found)")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching PubMed abstracts: {e}")
    except ET.ParseError as e:
        print(f"Error parsing PubMed XML response: {e}")
        # print(f"Response content causing parse error:\n{response.text[:500]}...") # Optional: print problematic content
    except Exception as e:
        print(f"An unexpected error occurred during abstract fetching: {e}")

    print(f"Successfully fetched abstracts for {len(papers)} papers.")
    return papers

# --- DeepSeek API 函数 ---
def summarize_text_deepseek(text, api_key):
    """使用 DeepSeek API 总结文本"""
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found.")
        return "Error: API key not configured."
    if not text:
        return "No text provided for summarization."

    print("Summarizing text with DeepSeek...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # 构建更明确的总结指令
    prompt = (
        "Please provide a concise summary (around 2-3 sentences) "
        "of the following academic paper abstract, focusing on the key findings and relevance "
        "to pharmacy or pharmaceutical sciences:\n\n"
        f"{text}"
    )
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert assistant specializing in summarizing biomedical research papers for pharmacists and pharmaceutical scientists."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200, # 稍微增加 token 限制以获得更好的总结
        "temperature": 0.3, # 降低温度以获得更集中、事实性的总结
        "stream": False # 确保获取完整响应
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60) # Add timeout
        response.raise_for_status()
        result = response.json()

        # 检查响应结构是否符合预期
        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            summary = result["choices"][0]["message"]["content"].strip()
            # 简单后处理，移除可能的引用标记
            summary = summary.replace("[...]", "").strip()
            print("Summarization successful.")
            return summary
        else:
            print("Error: Unexpected response format from DeepSeek API.")
            print("Response:", result)
            return "Error: Could not get summary from API response."

    except requests.exceptions.Timeout:
        print("Error: DeepSeek API request timed out.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API: {e}")
        # 尝试打印更详细的错误信息（如果响应存在）
        error_details = ""
        if hasattr(e, 'response') and e.response is not None:
             error_details = f" Status Code: {e.response.status_code}, Response Text: {e.response.text[:200]}..."
        print(error_details)
        return f"Error: API request failed - {e}"
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return f"Error: An unexpected error occurred - {e}"

# --- 主逻辑 ---
if __name__ == "__main__":
    print("Starting daily paper fetching and summarization process...")
    start_time = datetime.now()

    # 1. 从环境变量获取 API Key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("FATAL ERROR: DEEPSEEK_API_KEY environment variable not set.")
        sys.exit(1) # 使用 sys.exit()

    # 2. 搜索 PubMed 获取最近指定天数的论文 ID
    paper_ids = search_pubmed(PUBMED_SEARCH_TERM, days=DAYS_TO_SEARCH, retmax=MAX_PAPERS_TO_FETCH * 2)

    if not paper_ids:
        print("No new papers found matching the criteria.")
        sys.exit(0)

    # 3. 获取论文摘要 (只获取需要的数量)
    papers_to_summarize_ids = paper_ids[:MAX_PAPERS_TO_FETCH]
    papers_data = fetch_pubmed_abstracts(papers_to_summarize_ids)

    if not papers_data:
        print("Could not fetch abstracts for any of the selected papers.")
        sys.exit(0)

    # 4. 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 准备 Markdown 输出
    markdown_output = []
    today_date = start_time.strftime('%Y-%m-%d')
    markdown_output.append(f"# Daily Pharmacy Papers Summary - {today_date}")
    markdown_output.append(f"Fetched and summarized {len(papers_data)} papers on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    markdown_output.append("\n---") # 添加分隔符

    # 6. 遍历论文、总结并格式化输出
    summarized_count = 0
    for pmid, data in papers_data.items():
        print(f"\nProcessing PMID: {pmid}...")
        print(f"Title: {data['title']}")

        summary = summarize_text_deepseek(data['abstract'], deepseek_api_key)

        # 检查总结是否成功
        if not summary.startswith("Error:"):
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            markdown_output.append(f"\n## [{data['title']}]({pubmed_url})") # 标题作为二级标题，链接到 PubMed
            markdown_output.append(f"**PMID:** {pmid}")
            markdown_output.append(f"**Link:** [{pubmed_url}]({pubmed_url})")
            markdown_output.append(f"**Summary:**\n{summary}")
            markdown_output.append("\n---") # Paper separator
            summarized_count += 1
            print(f"Summary generated for PMID: {pmid}")
        else:
            print(f"Skipping PMID {pmid} due to summarization error: {summary}")
            # Optionally log the error or include a note in the output
            markdown_output.append(f"\n## {data['title']} (PMID: {pmid})")
            markdown_output.append(f"**Error:** Failed to generate summary. Reason: {summary}")
            markdown_output.append("\n---")

    # 7. 保存到 Markdown 文件
    output_filename = os.path.join(OUTPUT_DIR, f"daily_papers_{today_date}.md")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_output))
        print(f"\nSuccessfully generated summaries for {summarized_count} papers.")
        print(f"Output saved to: {output_filename}")
    except IOError as e:
        print(f"Error writing output file {output_filename}: {e}")
        # Optionally print the markdown to console as a fallback
        print("\n--- Markdown Output Fallback ---")
        print("\n".join(markdown_output))
        print("--- End Fallback ---")
        sys.exit(1)

    end_time = datetime.now()
    print(f"\nProcess finished in {end_time - start_time}.")
    sys.exit(0) # Explicitly exit with success code
