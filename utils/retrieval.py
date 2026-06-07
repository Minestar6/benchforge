"""Wikipedia 检索工具函数。"""

import hashlib
import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from benchforge.schemas import SourceDocument, DocumentStatus

# 遇到这些 h2 标题时停止正文提取（参考 wiki.py search_step）
_STOP_SECTIONS = {"references", "notes", "bibliography", "external links", "see also"}


def _get_with_retry(url: str, headers: dict, timeout: int, params: dict | None = None) -> requests.Response:
    """带 429 退避重试的 GET 请求，最多重试 4 次。"""
    backoff = 5
    for attempt in range(4):
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        print(f"  [429] rate limited, retrying in {backoff}s (attempt {attempt + 1}/4)...")
        time.sleep(backoff)
        backoff *= 2
    resp.raise_for_status()
    return resp


@dataclass
class WikipediaSearchResult:
    """Wikipedia 搜索结果。"""
    title: str
    url: str


def _generate_document_id(url: str) -> str:
    hash_obj = hashlib.sha256(url.encode('utf-8'))
    return f"doc_{hash_obj.hexdigest()[:12]}"


def _clean_str(text: str) -> str:
    """修正 unicode 乱码（同 wiki.py clean_str）。"""
    try:
        return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
    except Exception:
        return text


def get_pageviews(
    page_title: str,
    start_date: str = "2022010100",
    end_date: str = "2025010100",
) -> int:
    """查询 Wikimedia 某页面在时间段内的累计访问量。

    Args:
        page_title: 页面标题（空格用下划线替换）
        start_date / end_date: YYYYMMDDHR 格式

    Returns:
        累计访问量；失败时返回 0
    """
    title = page_title.replace(" ", "_")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
        f"/en.wikipedia/all-access/all-agents/{title}/daily/{start_date}/{end_date}"
    )
    headers = {"User-Agent": "BenchForge/0.1.0"}
    try:
        resp = _get_with_retry(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return sum(item["views"] for item in data.get("items", []))
    except Exception:
        pass
    return 0


def search_wikipedia(
    query: str,
    language: str = "en",
    max_pages: int = 5,
    request_timeout: int = 10,
    saliency_rerank: bool = False,
    saliency_top_k: int = 3,
    saliency_start_date: str = "2022010100",
    saliency_end_date: str = "2025010100",
) -> list[WikipediaSearchResult]:
    """搜索 Wikipedia 页面，可选按访问量筛选显著页面。

    当 saliency_rerank=True 时：
    - 内部搜索 saliency_top_k×2 个候选（无需用户配置候选池大小）
    - 调用 Wikimedia pageviews API 评分
    - 仅保留访问量最高的 saliency_top_k 个页面
    当 saliency_rerank=False 时：
    - 直接搜索并返回 max_pages 个结果
    """
    search_limit = saliency_top_k * 2 if saliency_rerank else max_pages

    results = []
    try:
        api_url = f"https://{language}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": search_limit,
            "format": "json",
        }
        headers = {"User-Agent": "BenchForge/0.1.0"}
        response = _get_with_retry(api_url, headers=headers, timeout=request_timeout, params=params)
        data = response.json()

        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            url = f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append(WikipediaSearchResult(title=title, url=url))

    except Exception as e:
        print(f"Wikipedia search failed: {e}")
        return results

    if not saliency_rerank or len(results) <= saliency_top_k:
        return results

    # 按 Wikimedia 访问量降序保留 top-k
    scored: list[tuple[WikipediaSearchResult, int]] = []
    for r in results:
        views = get_pageviews(r.title, saliency_start_date, saliency_end_date)
        print(f"  [saliency] {r.title}: {views / 1_000_000:.2f}M views")
        scored.append((r, views))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored[:saliency_top_k]]


def fetch_wikipedia_page(
    result: WikipediaSearchResult,
    run_id: str,
    language: str = "en",
    request_timeout: int = 15,
    min_paragraph_tokens: int = 20,
) -> SourceDocument:
    """抓取 Wikipedia 页面内容（使用 Parse API，避免 JS 渲染问题）。

    改进：停在 References/See Also 等参考文献区之前，过滤短段落，
    保留 h2/h3 标题作为结构标记（参考 wiki.py search_step）。
    """
    document_id = _generate_document_id(result.url)
    headers = {"User-Agent": "BenchForge/0.1.0"}

    # 从 URL 提取页面标题，用于 Parse API
    page_title = result.url.split("/wiki/")[-1].replace("_", " ")
    lang = language or "en"
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    try:
        response = _get_with_retry(
            api_url,
            headers=headers,
            timeout=request_timeout,
            params={"action": "parse", "page": page_title, "prop": "text", "format": "json"},
        )
        data = response.json()

        if "error" in data:
            raise ValueError(data["error"].get("info", "Parse API error"))

        html = data["parse"]["text"]["*"]
        title_text = data["parse"].get("title", result.title)
        soup = BeautifulSoup(html, "html.parser")

        lines: list[str] = []
        lead_parts: list[str] = []   # 第一个 h2 之前的段落（Wikipedia 导言）
        in_lead = True

        for tag in soup.find_all(["h2", "h3", "p", "ul"]):
            # 遇到参考文献区标题则停止
            if tag.name == "h2":
                in_lead = False      # 第一个 h2 标志导言结束
                heading = tag.get_text().strip().lower()
                # 去掉 [edit] 等尾缀
                heading = re.sub(r"\[.*?\]", "", heading).strip()
                if any(stop in heading for stop in _STOP_SECTIONS):
                    break
                lines.append(f"#Subheading: {tag.get_text().strip()}")
                continue

            if tag.name == "h3":
                lines.append(f"##Subheading: {tag.get_text().strip()}")
                continue

            text = tag.get_text().strip()
            # 过滤过短、句子不完整的段落
            if len(text.split()) < min_paragraph_tokens or text.count(".") < 1:
                continue

            text = _clean_str(text)
            # 去掉脚注引用标记 [1] [2]
            text = re.sub(r"\[\d+\]", "", text).strip()
            if not text:
                continue
            if text[-1] not in (".", "?", "!"):
                text += "."

            # 导言段只进 summary，不进 content
            if in_lead:
                lead_parts.append(text)
            else:
                lines.append(text)

        # 无 h2 时（消歧义页、短页）：前两段为 summary，其余为 content
        if in_lead and lead_parts:
            summary = "\n\n".join(lead_parts[:2])
            lines = lead_parts[2:]
        else:
            summary = "\n\n".join(lead_parts)

        content = "\n".join(lines)

        return SourceDocument(
            document_id=document_id,
            run_id=run_id,
            topic=result.title,
            language=language,
            title=title_text,
            url=result.url,
            summary=summary,
            content=content,
            status=DocumentStatus.FETCHED,
        )

    except Exception as e:
        return SourceDocument(
            document_id=document_id,
            run_id=run_id,
            topic=result.title,
            language=language,
            title=result.title,
            url=result.url,
            summary="",
            content="",
            metadata={"error": str(e)},
            status=DocumentStatus.FAILED,
        )
