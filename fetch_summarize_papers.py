import requests
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sys
import re

# Broadened search term for wider scope and newer trends
PUBMED_SEARCH_TERM = "network pharmacology OR pharmacy OR pharmaceutical sciences OR drug discovery OR drug development OR clinical pharmacy OR pharmacogenomics OR nanomedicine OR biotechnology drugs OR personalized medicine OR drug interactions OR traditional Chinese medicine OR pharmacokinetics OR pharmacodynamics OR drug delivery OR drug formulation OR drug metabolism OR drug safety OR drug efficacy OR drug design OR drug repurposing OR drug resistance OR drug toxicity OR drug side effects OR drug therapy OR drug monitoring OR drug regulation OR drug policy"
MAX_PAPERS_TO_FETCH = 20
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions" # Corrected API endpoint
DEEPSEEK_MODEL = "deepseek-chat"
OUTPUT_DIR = "output"
ARCHIVE_DIR = "archives" # Added archive directory constant
DAYS_TO_SEARCH = 1
README_FILE = "README.md"
README_START_MARKER = "<!-- DAILY_PAPERS_START -->"
README_END_MARKER = "<!-- DAILY_PAPERS_END -->"
# Added markers for the daily quote
README_QUOTE_START_MARKER = "<!-- DAILY_QUOTE_START -->"
README_QUOTE_END_MARKER = "<!-- DAILY_QUOTE_END -->"

def search_pubmed(query, days=DAYS_TO_SEARCH, retmax=MAX_PAPERS_TO_FETCH * 2):
    print(f"Searching PubMed for '{query}' in the last {days} day(s)...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_query = f"(\"{start_date.strftime('%Y/%m/%d')}\"[Date - Publication] : \"{end_date.strftime('%Y/%m/%d')}\"[Date - Publication])"

    params = {
        "db": "pubmed",
        "term": f"({query}) AND {date_query}",
        "retmode": "json",
        "retmax": retmax,
        "sort": "pub+date"
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
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

        root = ET.fromstring(response.content)
        for article in root.findall('.//PubmedArticle'):
            pmid_element = article.find('.//PMID')
            pmid = pmid_element.text if pmid_element is not None else None
            if not pmid: continue

            title_element = article.find('.//ArticleTitle')
            title = "".join(title_element.itertext()).strip() if title_element is not None else "No Title Available"

            abstract_text = ""
            abstract_elements = article.findall('.//Abstract/AbstractText')
            if abstract_elements:
                abstract_text = "\n\n".join(
                    "".join(part.itertext()).strip() for part in abstract_elements if "".join(part.itertext()).strip()
                )

            if abstract_text:
                papers[pmid] = {"title": title, "abstract": abstract_text}
                print(f"Fetched abstract for PMID: {pmid}")
            else:
                print(f"Skipping PMID: {pmid} (No abstract found)")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching PubMed abstracts: {e}")
    except ET.ParseError as e:
        print(f"Error parsing PubMed XML response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during abstract fetching: {e}")

    print(f"Successfully fetched abstracts for {len(papers)} papers.")
    return papers

def summarize_text_deepseek(text, api_key):
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
    prompt = (
        "请提供以下学术论文摘要的简明中文总结（约3-5句话），"
        "重点关注其关键发现及其与药学或药物科学的相关性,请突出创新性和启发性的研究内容\n\n"
        f"{text}"
    )
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位专门为药剂师和药物科学家总结生物医学研究论文的专家助手。请用中文回答。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300, # Increased max_tokens slightly for Chinese output
        "temperature": 0.3,
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            summary = result["choices"][0]["message"]["content"].strip()
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
        error_details = ""
        if hasattr(e, 'response') and e.response is not None:
             error_details = f" Status Code: {e.response.status_code}, Response Text: {e.response.text[:200]}..."
        print(error_details)
        return f"Error: API request failed - {e}"
    except Exception as e:
        print(f"An unexpected error occurred during summarization: {e}")
        return f"Error: An unexpected error occurred - {e}"

def generate_classical_quote(api_key):
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found.")
        return "Error: API key not configured."

    print("Generating classical quote with DeepSeek...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt = (
        "请创作一句关于知识、发现或日常学习的古典风格优美短句，并提供英文翻译。"
        "请确保句子简洁、典雅。格式如下：\n"
        "中文佳句\n"
        "English Translation"
    )
    data = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "你是一位精通中英文古典文学的AI助手。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7, # Slightly higher temperature for creativity
        "stream": False
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            quote = result["choices"][0]["message"]["content"].strip()
            print("Quote generation successful.")
            if "\n" in quote:
                return quote
            else:
                print("Warning: Generated quote format might be incorrect. Using as is.")
                return quote
        else:
            print("Error: Unexpected response format from DeepSeek API for quote generation.")
            print("Response:", result)
            return "Error: Could not get quote from API response."

    except requests.exceptions.Timeout:
        print("Error: DeepSeek API request for quote timed out.")
        return "Error: API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"Error calling DeepSeek API for quote: {e}")
        error_details = ""
        if hasattr(e, 'response') and e.response is not None:
             error_details = f" Status Code: {e.response.status_code}, Response Text: {e.response.text[:200]}..."
        print(error_details)
        return f"Error: API request failed - {e}"
    except Exception as e:
        print(f"An unexpected error occurred during quote generation: {e}")
        return f"Error: An unexpected error occurred - {e}"

def update_readme(readme_path, start_marker, end_marker, new_content):
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()

        pattern = re.compile(f"({re.escape(start_marker)}\s*).*?(\s*{re.escape(end_marker)})", re.DOTALL)
        replacement_content = f"{start_marker}\n\n{new_content}\n\n{end_marker}"

        updated_readme, num_replacements = pattern.subn(replacement_content, readme_content)

        if num_replacements == 0:
            print(f"Warning: Markers '{start_marker}' and '{end_marker}' not found in {readme_path}. README not updated.")
            return False

        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(updated_readme)
        print(f"Successfully updated {readme_path}")
        return True

    except FileNotFoundError:
        print(f"Error: README file not found at {readme_path}")
        return False
    except IOError as e:
        print(f"Error reading or writing README file {readme_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during README update: {e}")
        return False

def save_archive(archive_dir, date_str, content):
    """Saves the daily summary content to an archive file."""
    try:
        os.makedirs(archive_dir, exist_ok=True)
        archive_filename = os.path.join(archive_dir, f"{date_str}.md")
        with open(archive_filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully saved archive to {archive_filename}")
        return True
    except IOError as e:
        print(f"Error writing archive file {archive_filename}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during archive saving: {e}")
        return False

if __name__ == "__main__":
    print("Starting daily paper fetching and summarization process...")
    start_time = datetime.now()

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("FATAL ERROR: DEEPSEEK_API_KEY environment variable not set.")
        sys.exit(1)

    # Generate the daily quote
    daily_quote = generate_classical_quote(deepseek_api_key)
    if daily_quote.startswith("Error:"):
        print(f"Failed to generate daily quote: {daily_quote}. Proceeding without quote.")
        daily_quote_content = "*Failed to generate daily quote.*"
    else:
        # Attempt to split into Chinese and English parts, stripping whitespace and removing empty lines
        quote_lines = [line.strip() for line in daily_quote.strip().split('\n') if line.strip()]

        if len(quote_lines) >= 2:
            # Assume first line is Chinese, second is English
            chinese_quote = quote_lines[0]
            english_translation = quote_lines[1]
            daily_quote_content = f"*{chinese_quote}* / *{english_translation}*"
            print(f"Formatted quote with translation: {daily_quote_content}")
        else:
            # Fallback if format is unexpected (less than 2 non-empty lines)
            print(f"Warning: Quote format unexpected. Expected 'Chinese\nEnglish', received: '{daily_quote}'. Using raw output.")
            # Display only the received content without the ' / ' separator
            daily_quote_content = f"*{daily_quote.strip()}*" # Use the original stripped quote

    quote_update_successful = update_readme(
        README_FILE,
        README_QUOTE_START_MARKER,
        README_QUOTE_END_MARKER,
        daily_quote_content
    )

    paper_ids = search_pubmed(PUBMED_SEARCH_TERM, days=DAYS_TO_SEARCH, retmax=MAX_PAPERS_TO_FETCH * 2)
    today_date = start_time.strftime('%Y-%m-%d')
    readme_update_successful = False # Initialize flag

    if not paper_ids:
        print("No new papers found matching the criteria. Exiting.")
        no_papers_content = f"### Summary for {today_date}\n*No new papers found for '{PUBMED_SEARCH_TERM}' on {today_date}.*"
        # Update README with no papers message
        readme_update_successful = update_readme(
            README_FILE,
            README_START_MARKER,
            README_END_MARKER,
            no_papers_content
        )
        # Also save the "no papers" message to archive
        save_archive(ARCHIVE_DIR, today_date, no_papers_content)
        print(f"\nProcess finished in {datetime.now() - start_time}.")
        # Exit early if no papers found
        sys.exit(0 if (quote_update_successful and readme_update_successful) else 1)
    else:
        # Papers were found, proceed with fetching and summarizing
        papers = fetch_pubmed_abstracts(paper_ids[:MAX_PAPERS_TO_FETCH]) # Limit fetching

        if not papers:
            print("Could not fetch abstracts for any papers. Exiting.")
            no_abstracts_content = f"### Summary for {today_date}\n*Found paper IDs but could not fetch abstracts.*"
            readme_update_successful = update_readme(
                README_FILE,
                README_START_MARKER,
                README_END_MARKER,
                no_abstracts_content
            )
            save_archive(ARCHIVE_DIR, today_date, no_abstracts_content)
            print(f"\nProcess finished in {datetime.now() - start_time}.")
            sys.exit(1) # Exit with error if abstracts couldn't be fetched

        # Initialize markdown list and counter *before* the loop
        markdown_output_for_readme = [f"### Summary for {today_date}"]
        markdown_output_for_readme.append(f"*Fetched and summarized {len(papers)} papers matching '{PUBMED_SEARCH_TERM}'.*")
        markdown_output_for_readme.append("") # Add a blank line
        summarized_count = 0

        # Loop through fetched papers
        for pmid, data in papers.items():
            print(f"\nProcessing PMID: {pmid}...")
            print(f"Title: {data['title']}")

            summary = summarize_text_deepseek(data['abstract'], deepseek_api_key)

            if not summary.startswith("Error:"):
                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                markdown_output_for_readme.append(f"*   **[{data['title']}]({pubmed_url})** (PMID: {pmid})")
                summary_lines = summary.split('\n')
                for line in summary_lines:
                    markdown_output_for_readme.append(f"    *   {line.strip()}")
                markdown_output_for_readme.append("")
                summarized_count += 1
                print(f"Summary generated for PMID: {pmid}")
            else:
                print(f"Skipping PMID {pmid} due to summarization error: {summary}")
                # Append error message to README for this paper
                markdown_output_for_readme.append(f"*   **{data['title']}** (PMID: {pmid}) - *Error generating summary: {summary}*")
                markdown_output_for_readme.append("")

        # Now update the README with the generated summaries
        readme_update_successful = update_readme(
            README_FILE,
            README_START_MARKER,
            README_END_MARKER,
            "\n".join(markdown_output_for_readme)
        )

        # Save the successful summary to archive
        archive_content = "\n".join(markdown_output_for_readme)
        save_archive(ARCHIVE_DIR, today_date, archive_content)

        end_time = datetime.now()
        print(f"\nSuccessfully generated summaries for {summarized_count} papers.")
        print(f"Process finished in {end_time - start_time}.")
        sys.exit(0 if (quote_update_successful and readme_update_successful) else 1)
