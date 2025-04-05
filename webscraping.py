import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from readability import Document
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm
from webdriver_manager.microsoft import EdgeChromiumDriverManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapedContent:
    """Data structure for storing scraped content with metadata."""
    url: str
    title: str
    html: str
    text: str
    tables: List[Dict[str, Union[str, List[List[str]]]]] = field(default_factory=list)
    lists: List[Dict[str, Union[str, List[str]]]] = field(default_factory=list)
    forms: List[Dict[str, Union[str, Dict[str, str]]]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    headings: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        """Convert the content to a dictionary format."""
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "tables": self.tables,
            "lists": self.lists,
            "forms": self.forms,
            "links": self.links,
            "images": self.images,
            "headings": dict(self.headings),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def to_rag_format(self) -> Dict:
        """
        Convert the content to a RAG-friendly format suitable for vectorization.
        This format creates structured chunks with appropriate context and metadata.
        """
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
                    **self.metadata
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


class WebScraper:
    """Advanced web scraper for extracting structured content from websites."""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str = "scraped_data",
        max_pages: int = 100,
        delay: float = 1.0,
        headless: bool = True,
        use_selenium: bool = True,
        threads: int = 4,
        respect_robots: bool = True,
        user_agent: str = "RAGScraper/1.0",
    ):
        """
        Initialize the web scraper.
        
        Args:
            base_url: The starting URL for the scraper
            output_dir: Directory to store scraped data
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds
            headless: Whether to run the browser in headless mode
            use_selenium: Whether to use Selenium for JavaScript rendering
            threads: Number of threads for parallel scraping
            respect_robots: Whether to respect robots.txt
            user_agent: User agent string to use for requests
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay
        self.headless = headless
        self.use_selenium = use_selenium
        self.threads = threads
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up tracked URLs
        self.visited_urls: Set[str] = set()
        self.queued_urls: Set[str] = set([base_url])
        self.disallowed_patterns: List[str] = []
        
        # Extract domain to stay within the same domain
        self.domain = urlparse(base_url).netloc
        
        # Set up the Selenium driver if needed
        self.driver = None
        if use_selenium:
            self._setup_selenium()
        
        # Check robots.txt
        if respect_robots:
            self._parse_robots_txt()

    def _setup_selenium(self):
        """Set up the Selenium WebDriver using Microsoft Edge."""
        edge_options = Options()
        if self.headless:
            edge_options.add_argument("--headless")
        
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--window-size=1920,1080")
        edge_options.add_argument(f"user-agent={self.user_agent}")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--no-sandbox")
        
        service = Service(EdgeChromiumDriverManager().install())
        self.driver = webdriver.Edge(service=service, options=edge_options)
        
    def _parse_robots_txt(self):
        """Parse robots.txt to respect disallowed paths."""
        robots_url = urljoin(self.base_url, "/robots.txt")
        try:
            response = requests.get(robots_url, headers={"User-Agent": self.user_agent})
            if response.status_code == 200:
                for line in response.text.split("\n"):
                    if "Disallow:" in line and not line.startswith("#"):
                        disallowed = line.split("Disallow:")[1].strip()
                        if disallowed:
                            self.disallowed_patterns.append(disallowed)
                            
            logger.info(f"Found {len(self.disallowed_patterns)} disallowed patterns in robots.txt")
        except Exception as e:
            logger.warning(f"Error parsing robots.txt: {e}")
    
    def _is_url_allowed(self, url: str) -> bool:
        """Check if a URL is allowed to be scraped based on robots.txt."""
        if not self.respect_robots or not self.disallowed_patterns:
            return True
        
        path = urlparse(url).path
        for pattern in self.disallowed_patterns:
            if pattern == "/" and path == "/":
                return False
            if pattern.endswith("*") and path.startswith(pattern[:-1]):
                return False
            if path == pattern:
                return False
        
        return True
    
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
                and self._is_url_allowed(absolute_url)
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
                clean_url = urlparse("").geturl()
                
                urls.append(clean_url)
                self.queued_urls.add(clean_url)
        
        return urls
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Union[str, List[List[str]]]]]:
        """Extract tables from the webpage."""
        tables = []
        
        for table in soup.find_all("table"):
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
            
            table_data.append(headers if headers else [""] * len(table.find_all("tr")[0].find_all(["td", "th"])))
            
            # Extract rows
            for row in table.find_all("tr"):
                # Skip header row if we've already processed it
                if row == table.find("tr") and not header_row:
                    continue
                
                row_data = []
                for cell in row.find_all(["td", "th"]):
                    row_data.append(cell.get_text(strip=True))
                
                if row_data:
                    table_data.append(row_data)
            
            tables.append({
                "caption": caption,
                "data": table_data,
            })
        
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
        
        # Process description lists
        for dl in soup.find_all("dl"):
            items = []
            current_term = ""
            
            for child in dl.children:
                if child.name == "dt":
                    current_term = child.get_text(strip=True)
                elif child.name == "dd" and current_term:
                    items.append(f"{current_term}: {child.get_text(strip=True)}")
                    current_term = ""
            
            if items:
                lists.append({
                    "title": "",
                    "items": items,
                    "type": "dl"
                })
        
        return lists
    
    def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """Extract forms and their fields from the webpage."""
        forms = []
        
        for form in soup.find_all("form"):
            form_data = {
                "action": form.get("action", ""),
                "method": form.get("method", ""),
                "fields": {}
            }
            
            # Extract form title/heading if available
            form_title = ""
            heading = form.find(["h1", "h2", "h3", "h4", "h5", "h6", "legend"])
            if heading:
                form_title = heading.get_text(strip=True)
            form_data["title"] = form_title
            
            # Extract form fields
            fields = {}
            
            for input_field in form.find_all(["input", "textarea", "select"]):
                field_name = input_field.get("name", "")
                field_type = input_field.get("type", "text") if input_field.name == "input" else input_field.name
                field_label = ""
                
                # Try to find a label for this field
                if input_field.get("id"):
                    label = soup.find("label", attrs={"for": input_field["id"]})
                    if label:
                        field_label = label.get_text(strip=True)
                
                if field_name:
                    fields[field_name] = {
                        "type": field_type,
                        "label": field_label
                    }
            
            form_data["fields"] = fields
            
            if fields:
                forms.append(form_data)
        
        return forms
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract images and their metadata from the webpage."""
        images = []
        
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if src:
                # Convert relative URLs to absolute
                src = urljoin(base_url, src)
                
                alt = img.get("alt", "")
                title = img.get("title", "")
                
                # Try to get figure caption if image is inside a figure
                caption = ""
                figure_parent = img.find_parent("figure")
                if figure_parent:
                    fig_caption = figure_parent.find("figcaption")
                    if fig_caption:
                        caption = fig_caption.get_text(strip=True)
                
                images.append({
                    "src": src,
                    "alt": alt,
                    "title": title,
                    "caption": caption
                })
        
        return images
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings by level from the webpage."""
        headings = defaultdict(list)
        
        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text:
                    headings[f"h{level}"].append(text)
        
        return headings
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata from the webpage."""
        metadata = {}
        
        # Extract meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "")
            property_val = meta.get("property", "")
            content = meta.get("content", "")
            
            if name and content:
                metadata[name] = content
            elif property_val and content:
                metadata[property_val] = content
        
        return metadata
    
    def _get_page_content(self, url: str) -> Optional[ScrapedContent]:
        """
        Fetch and parse a single page using requests or Selenium.
        
        Args:
            url: The URL to scrape
            
        Returns:
            ScrapedContent object or None if failed
        """
        try:
            if self.use_selenium:
                self.driver.get(url)
                # Wait for page to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                
                # Additional wait for dynamic content
                time.sleep(2)
                
                # Get the page source
                html = self.driver.page_source
                soup = BeautifulSoup(html, "html.parser")
                title = self.driver.title
            else:
                response = requests.get(
                    url, 
                    headers={"User-Agent": self.user_agent},
                    timeout=30
                )
                response.raise_for_status()
                html = response.text
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.text if soup.title else url
            
            # Use Readability for better text extraction
            doc = Document(html)
            article_html = doc.summary()
            article_soup = BeautifulSoup(article_html, "html.parser")
            
            # Clean text by removing extra whitespace
            text = ' '.join(article_soup.get_text().split())
            
            # Extract structured content
            tables = self._extract_tables(soup)
            lists = self._extract_lists(soup)
            forms = self._extract_forms(soup)
            images = self._extract_images(soup, url)
            headings = self._extract_headings(soup)
            metadata = self._extract_metadata(soup)
            
            # Extract new URLs
            new_urls = self._extract_urls(soup, url)
            
            # Create scraped content object
            content = ScrapedContent(
                url=url,
                title=title.strip() if title else url,
                html=html,
                text=text,
                tables=tables,
                lists=lists,
                forms=forms,
                links=[{"url": u} for u in new_urls],
                images=images,
                headings=headings,
                metadata=metadata,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            return content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
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
        
        # Close Selenium driver if used
        if self.driver:
            self.driver.quit()
        
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
                            html="",  # Don't need to load HTML here
                            text=data["text"],
                            tables=data.get("tables", []),
                            lists=data.get("lists", []),
                            forms=data.get("forms", []),
                            links=data.get("links", []),
                            images=data.get("images", []),
                            headings=defaultdict(list, data.get("headings", {})),
                            metadata=data.get("metadata", {}),
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
    parser = argparse.ArgumentParser(description="Advanced web scraper for RAG applications")
    parser.add_argument("url", help="The website URL to scrape")
    parser.add_argument("--output", "-o", default="scraped_data", help="Output directory")
    parser.add_argument("--max-pages", "-m", type=int, default=100, help="Maximum pages to scrape")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--no-selenium", action="store_false", dest="use_selenium", help="Don't use Selenium")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of threads for parallel scraping")
    parser.add_argument("--no-robots", action="store_false", dest="respect_robots", help="Don't respect robots.txt")
    parser.add_argument("--user-agent", "-u", default="RAGScraper/1.0", help="User agent string")
    
    args = parser.parse_args()
    
    # Create and run the scraper
    scraper = WebScraper(
        base_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
        delay=args.delay,
        headless=args.headless,
        use_selenium=args.use_selenium,
        threads=args.threads,
        respect_robots=args.respect_robots,
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
