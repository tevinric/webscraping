import argparse
import json
import logging
import os
import re
import time
import ssl
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm

# Configure SSL context to ignore certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Disable warnings for insecure requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ScrapedContent:
    """Data structure for storing scraped content with metadata."""
    url: str
    title: str
    text: str
    tables: List[Dict[str, Union[str, List[List[str]]]]] = field(default_factory=list)
    lists: List[Dict[str, Union[str, List[str]]]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    headings: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert the content to a dictionary format."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "tables": self.tables,
            "lists": self.lists,
            "links": self.links,
            "headings": dict(self.headings),
            "timestamp": self.timestamp,
        }

    def to_rag_format(self) -> List[Dict]:
        """Convert the content to a RAG-friendly format suitable for vectorization."""
        chunks = []
        
        # Main text chunk
        if self.text:
            chunks.append({
                "text": self.text[:10000],  # Limit size to avoid token limits
                "metadata": {
                    "url": self.url,
                    "title": self.title,
                    "type": "main_content",
                    "timestamp": self.timestamp,
                }
            })
        
        # Table chunks
        for i, table in enumerate(self.tables):
            table_content = f"Table: {table.get('caption', 'Unnamed Table')}\n"
            table_data = table.get('data', [])
            
            if table_data:
                # Convert table data to string representation
                df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                table_content += df.to_string(index=False)
                
                chunks.append({
                    "text": table_content,
                    "metadata": {
                        "url": self.url,
                        "title": self.title,
                        "type": "table",
                        "table_id": i,
                        "timestamp": self.timestamp,
                    }
                })
        
        # List chunks
        for i, list_item in enumerate(self.lists):
            list_content = f"List: {list_item.get('title', 'Unnamed List')}\n"
            list_content += "\n".join([f"• {item}" for item in list_item.get('items', [])])
            
            chunks.append({
                "text": list_content,
                "metadata": {
                    "url": self.url,
                    "title": self.title,
                    "type": "list",
                    "list_id": i,
                    "timestamp": self.timestamp,
                }
            })
                
        return chunks


class SimpleWebScraper:
    """Simplified web scraper for extracting structured content from websites."""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str = "scraped_data",
        max_pages: int = 100,
        delay: float = 1.0,
        threads: int = 4,
        user_agent: str = "RAGScraper/1.0",
    ):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The starting URL for the scraper
            output_dir: Directory to store scraped data
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds
            threads: Number of threads for parallel scraping
            user_agent: User agent string to use for requests
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay
        self.threads = threads
        self.user_agent = user_agent
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up tracked URLs
        self.visited_urls: Set[str] = set()
        self.queued_urls: Set[str] = set([base_url])
        
        # Extract domain to stay within the same domain
        self.domain = urlparse(base_url).netloc
        
        # Set up a session with default configurations
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        # Disable SSL verification globally for this session
        self.session.verify = False
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if a URL belongs to the same domain as the base URL."""
        parsed_url = urlparse(url)
        return parsed_url.netloc == self.domain
    
    def _extract_urls(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all URLs from a page and filter them."""
        urls = []
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Handle relative URLs
            absolute_url = urljoin(current_url, href)
            
            # Filter URLs
            if (
                self._is_same_domain(absolute_url)
                and absolute_url not in self.visited_urls
                and absolute_url not in self.queued_urls
                # Avoid common non-content URLs
                and not any(pattern in absolute_url for pattern in 
                            ["/login", "/logout", "/signup", "/register", "#"])
                # Avoid duplicate content with trailing slashes
                and not (absolute_url.rstrip("/") in self.visited_urls
                         or absolute_url.rstrip("/") + "/" in self.visited_urls)
            ):
                # Remove fragments
                url_parts = list(urlparse(absolute_url))
                url_parts[5] = ""  # Remove fragment
                clean_url = urljoin(absolute_url, "")
                
                urls.append(clean_url)
                self.queued_urls.add(clean_url)
        
        return urls
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Union[str, List[List[str]]]]]:
        """Extract tables from the webpage."""
        tables = []
        
        for table in soup.find_all("table"):
            try:
                table_data = []
                
                # Extract caption/title
                caption = ""
                caption_tag = table.find("caption")
                if caption_tag:
                    caption = caption_tag.get_text(strip=True)
                
                # Extract headers
                headers = []
                header_row = table.find("thead")
                if header_row:
                    for th in header_row.find_all("th"):
                        headers.append(th.get_text(strip=True))
                else:
                    # Try finding headers in the first row if no thead
                    first_row = table.find("tr")
                    if first_row:
                        for th in first_row.find_all(["th", "td"]):
                            headers.append(th.get_text(strip=True))
                
                # Add headers to table data
                if headers:
                    table_data.append(headers)
                elif table.find("tr"):
                    # If no headers, use empty strings based on first row cell count
                    first_row_cells = table.find("tr").find_all(["td", "th"])
                    if first_row_cells:
                        table_data.append([""] * len(first_row_cells))
                
                # Extract rows
                for row in table.find_all("tr"):
                    # Skip header row if we've already processed it
                    if row == table.find("tr") and not header_row and headers:
                        continue
                    
                    row_data = []
                    for cell in row.find_all(["td", "th"]):
                        row_data.append(cell.get_text(strip=True))
                    
                    if row_data:
                        table_data.append(row_data)
                
                if len(table_data) > 0:
                    tables.append({
                        "caption": caption,
                        "data": table_data,
                    })
            except Exception as e:
                logger.warning(f"Error extracting table: {e}")
        
        return tables
    
    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Union[str, List[str]]]]:
        """Extract lists from the webpage."""
        lists = []
        
        # Function to extract list items
        def extract_list_items(list_element):
            items = []
            for li in list_element.find_all("li", recursive=False):
                text = li.get_text(strip=True)
                if text:
                    items.append(text)
            return items
        
        # Process ordered and unordered lists
        for list_tag in soup.find_all(["ul", "ol"]):
            try:
                # Skip nested lists (only process top-level lists)
                if list_tag.parent and list_tag.parent.name in ["li", "ul", "ol"]:
                    continue
                
                # Try to find a heading before the list
                title = ""
                prev_sibling = list_tag.find_previous_sibling(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
                if prev_sibling:
                    title = prev_sibling.get_text(strip=True)
                
                items = extract_list_items(list_tag)
                
                if items:
                    lists.append({
                        "title": title,
                        "items": items,
                        "type": list_tag.name
                    })
            except Exception as e:
                logger.warning(f"Error extracting list: {e}")
        
        return lists
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings by level from the webpage."""
        headings = defaultdict(list)
        
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                try:
                    text = heading.get_text(strip=True)
                    if text:
                        headings[f"h{level}"].append(text)
                except Exception as e:
                    logger.warning(f"Error extracting heading: {e}")
        
        return headings
    
    def _get_page_content(self, url: str) -> Optional[ScrapedContent]:
        """
        Fetch and parse a single page.
        
        Args:
            url: The URL to scrape
            
        Returns:
            ScrapedContent object or None if failed
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Add retry delay if this is a retry
                if retry_count > 0:
                    time.sleep(self.delay * 2)
                
                # Make the request with session (SSL verification disabled)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                html = response.text
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.get_text(strip=True) if soup.title else url
                
                # Use Readability for better text extraction
                try:
                    doc = Document(html)
                    article_html = doc.summary()
                    article_soup = BeautifulSoup(article_html, "html.parser")
                    text = ' '.join(article_soup.get_text().split())
                except Exception as e:
                    logger.warning(f"Readability extraction failed: {e}. Using regular HTML parsing.")
                    text = ' '.join(soup.get_text().split())
                
                # Extract structured content
                tables = self._extract_tables(soup)
                lists = self._extract_lists(soup)
                headings = self._extract_headings(soup)
                
                # Extract new URLs
                new_urls = self._extract_urls(soup, url)
                
                # Create scraped content object
                content = ScrapedContent(
                    url=url,
                    title=title.strip() if title else url,
                    text=text,
                    tables=tables,
                    lists=lists,
                    links=[{"url": u} for u in new_urls],
                    headings=headings,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                return content
                
            except requests.exceptions.SSLError as e:
                logger.warning(f"SSL Error on {url} (retry {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request Error on {url} (retry {retry_count+1}/{max_retries}): {e}")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Unexpected error scraping {url}: {e}")
                return None
                
        logger.error(f"Failed to scrape {url} after {max_retries} retries")
        return None
    
    def _save_content(self, content: ScrapedContent):
        """Save scraped content to JSON file."""
        if not content:
            return
        
        # Generate filename from URL
        filename = re.sub(r'[^\w\-_]', '_', content.url)
        filename = re.sub(r'_+', '_', filename)
        filename = filename[-240:] if len(filename) > 250 else filename
        output_path = os.path.join(self.output_dir, f"{filename}.json")
        
        # Save content as JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(content.to_dict(), f, ensure_ascii=False, indent=2)
    
    def scrape(self):
        """
        Main method to scrape the website.
        
        Returns:
            dict: Mapping of URLs to their scraped content
        """
        result = {}
        progress_bar = tqdm(total=self.max_pages, desc="Scraping pages")
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            while self.queued_urls and len(self.visited_urls) < self.max_pages:
                # Get a batch of URLs to process
                batch_size = min(self.threads, self.max_pages - len(self.visited_urls))
                batch_urls = []
                
                for _ in range(batch_size):
                    if not self.queued_urls:
                        break
                    url = self.queued_urls.pop()
                    batch_urls.append(url)
                
                # Submit scraping tasks
                future_to_url = {
                    executor.submit(self._get_page_content, url): url
                    for url in batch_urls
                }
                
                # Process results as they complete
                for future in future_to_url:
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            result[url] = content
                            self._save_content(content)
                            self.visited_urls.add(url)
                            progress_bar.update(1)
                            
                            # Add a small delay between requests
                            time.sleep(self.delay)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {e}")
        
        progress_bar.close()
        
        # Save the index file with all URLs
        index = {
            "base_url": self.base_url,
            "total_pages": len(self.visited_urls),
            "pages": list(self.visited_urls)
        }
        
        with open(os.path.join(self.output_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
        
        return result

    def prepare_for_rag(self, output_file: str = "rag_data.json"):
        """
        Convert all scraped data to a format suitable for RAG applications.
        
        Args:
            output_file: Path to save the RAG-formatted data
            
        Returns:
            List of documents in RAG format
        """
        rag_docs = []
        
        # Read all scraped content
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json") and filename != "index.json" and filename != output_file:
                file_path = os.path.join(self.output_dir, filename)
                
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        content = ScrapedContent(
                            url=data["url"],
                            title=data["title"],
                            text=data["text"],
                            tables=data.get("tables", []),
                            lists=data.get("lists", []),
                            links=data.get("links", []),
                            headings=defaultdict(list, data.get("headings", {})),
                            timestamp=data.get("timestamp", "")
                        )
                        
                        # Convert to RAG format
                        rag_docs.extend(content.to_rag_format())
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
        
        # Save RAG-formatted data
        with open(os.path.join(self.output_dir, output_file), "w", encoding="utf-8") as f:
            json.dump(rag_docs, f, ensure_ascii=False, indent=2)
        
        return rag_docs


def main():
    """Main entry point for the scraper."""
    parser = argparse.ArgumentParser(description="Simple web scraper for RAG applications")
    parser.add_argument("url", help="The website URL to scrape")
    parser.add_argument("--output", "-o", default="scraped_data", help="Output directory")
    parser.add_argument("--max-pages", "-m", type=int, default=100, help="Maximum pages to scrape")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of threads for parallel scraping")
    parser.add_argument("--user-agent", "-u", default="RAGScraper/1.0", help="User agent string")
    
    args = parser.parse_args()
    
    # Create and run the scraper
    scraper = SimpleWebScraper(
        base_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
        delay=args.delay,
        threads=args.threads,
        user_agent=args.user_agent
    )
    
    print(f"Starting to scrape {args.url}")
    result = scraper.scrape()
    print(f"Scraped {len(result)} pages successfully")
    
    # Prepare data for RAG
    print("Preparing data for RAG application...")
    rag_docs = scraper.prepare_for_rag()
    print(f"Generated {len(rag_docs)} documents for RAG")
    print(f"RAG-ready data saved to {os.path.join(args.output, 'rag_data.json')}")


if __name__ == "__main__":
    main()
