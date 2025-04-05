import argparse
import json
import logging
import os
import re
import time
import ssl
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union, Any, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from readability import Document
from tqdm import tqdm

# Optional Selenium support for JavaScript-heavy pages
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

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
class ContentNode:
    """A node in the content hierarchy."""
    type: str  # 'heading', 'paragraph', 'list', 'table', etc.
    content: str
    level: int = 0  # For headings (h1-h6) or list nesting level
    children: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert the node to a dictionary."""
        return {
            "type": self.type,
            "content": self.content,
            "level": self.level,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    def to_text(self, indent: int = 0) -> str:
        """Convert the node and its children to formatted text."""
        text = " " * indent + self.content + "\n"
        for child in self.children:
            text += child.to_text(indent + 2)
        return text


@dataclass
class ScrapedContent:
    """Data structure for storing scraped content with metadata."""
    url: str
    title: str
    text: str
    content_tree: List[ContentNode] = field(default_factory=list)
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
            "content_tree": [node.to_dict() for node in self.content_tree],
            "tables": self.tables,
            "lists": self.lists,
            "links": self.links,
            "headings": dict(self.headings),
            "timestamp": self.timestamp,
        }
    
    def to_hierarchical_text(self) -> str:
        """Convert the content tree to hierarchical text."""
        text = f"Title: {self.title}\n\n"
        for node in self.content_tree:
            text += node.to_text()
        return text

    def to_rag_format(self) -> List[Dict]:
        """
        Convert the content to a RAG-friendly format suitable for vectorization,
        preserving the hierarchical structure.
        """
        chunks = []
        
        # Process the whole page for context
        chunks.append({
            "text": self.to_hierarchical_text(),
            "metadata": {
                "url": self.url,
                "title": self.title,
                "type": "full_content",
                "timestamp": self.timestamp,
            }
        })
        
        # Create a chunk for each major section (based on h1, h2)
        current_section = None
        current_section_title = ""
        current_section_content = []
        
        for node in self.content_tree:
            if node.type == 'heading' and node.level <= 2:
                # If we were building a section, add it to chunks
                if current_section_content and current_section_title:
                    section_text = f"# {current_section_title}\n\n"
                    for content in current_section_content:
                        section_text += content.to_text()
                    
                    chunks.append({
                        "text": section_text,
                        "metadata": {
                            "url": self.url,
                            "title": self.title,
                            "section": current_section_title,
                            "type": "section",
                            "timestamp": self.timestamp,
                        }
                    })
                
                # Start a new section
                current_section_title = node.content
                current_section_content = [node]
            elif current_section_title:
                # Add content to the current section
                current_section_content.append(node)
        
        # Add the last section if there's one being built
        if current_section_content and current_section_title:
            section_text = f"# {current_section_title}\n\n"
            for content in current_section_content:
                section_text += content.to_text()
            
            chunks.append({
                "text": section_text,
                "metadata": {
                    "url": self.url,
                    "title": self.title,
                    "section": current_section_title,
                    "type": "section",
                    "timestamp": self.timestamp,
                }
            })
        
        # Table chunks
        for i, table in enumerate(self.tables):
            table_content = f"Table: {table.get('caption', 'Unnamed Table')}\n"
            table_data = table.get('data', [])
            
            if table_data:
                # Convert table data to string representation
                try:
                    df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                    table_content += df.to_string(index=False)
                except Exception as e:
                    # Fallback if pandas conversion fails
                    for row in table_data:
                        table_content += " | ".join(str(cell) for cell in row) + "\n"
                
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


class EnhancedWebScraper:
    """Advanced web scraper for extracting structured content with hierarchy preservation."""
    
    def __init__(
        self,
        base_url: str,
        output_dir: str = "scraped_data",
        max_pages: int = 100,
        delay: float = 1.0,
        threads: int = 4,
        user_agent: str = "RAGScraper/1.0",
        use_selenium: bool = False,
        headless: bool = True,
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
            use_selenium: Whether to use Selenium for JavaScript rendering
            headless: Whether to run Selenium in headless mode
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.max_pages = max_pages
        self.delay = delay
        self.threads = threads
        self.user_agent = user_agent
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.headless = headless
        
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
        
        # Set up Selenium if needed
        self.driver = None
        if self.use_selenium:
            self._setup_selenium()
    
    def _setup_selenium(self):
        """Set up Selenium WebDriver."""
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium is not available. Install it with: pip install selenium webdriver-manager")
            return
        
        try:
            chrome_options = ChromeOptions()
            if self.headless:
                chrome_options.add_argument("--headless")
            
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(f"user-agent={self.user_agent}")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--disable-extensions")
            
            service = ChromeService(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logger.info("Selenium WebDriver initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {e}")
            self.use_selenium = False
    
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
                        # Ensure all rows have the same number of cells
                        if table_data and len(table_data[0]) > len(row_data):
                            row_data.extend([''] * (len(table_data[0]) - len(row_data)))
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
        
        # Function to extract list items recursively, handling nested lists
        def extract_list_items(list_element, depth=0):
            items = []
            for li in list_element.find_all("li", recursive=False):
                # Get the direct text of this list item (excluding nested lists)
                item_text = ""
                for content in li.contents:
                    if isinstance(content, NavigableString):
                        item_text += content.strip()
                    elif content.name not in ['ul', 'ol']:
                        item_text += content.get_text(strip=True)
                
                item_text = item_text.strip()
                if item_text:
                    # Add indentation based on depth for nested lists
                    items.append(("  " * depth) + item_text)
                
                # Process nested lists
                nested_lists = li.find_all(['ul', 'ol'], recursive=False)
                for nested_list in nested_lists:
                    nested_items = extract_list_items(nested_list, depth + 1)
                    items.extend(nested_items)
            
            return items
        
        # Process standard lists (ul, ol)
        for list_index, list_tag in enumerate(soup.find_all(["ul", "ol"])):
            try:
                # Skip if this list is nested within another list
                if list_tag.parent and list_tag.parent.name in ["li", "ul", "ol"]:
                    continue
                
                # Try to find a heading or paragraph before the list
                title = ""
                prev_elem = list_tag.find_previous_sibling(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
                if prev_elem:
                    title = prev_elem.get_text(strip=True)
                
                # Extract all items including nested ones
                items = extract_list_items(list_tag)
                
                if items:
                    lists.append({
                        "title": title,
                        "items": items,
                        "type": list_tag.name,
                        "position": list_index  # Keep track of position in document
                    })
            except Exception as e:
                logger.warning(f"Error extracting list: {e}")
        
        # Process definition lists (dl)
        for dl_index, dl in enumerate(soup.find_all("dl")):
            try:
                items = []
                for dt in dl.find_all("dt"):
                    term = dt.get_text(strip=True)
                    
                    # Find all dd elements until the next dt
                    definitions = []
                    dd = dt.find_next_sibling()
                    while dd and dd.name == "dd":
                        definitions.append(dd.get_text(strip=True))
                        dd = dd.find_next_sibling()
                    
                    # Add term and its definitions
                    if term:
                        items.append(term)
                        for definition in definitions:
                            items.append(f"  {definition}")
                
                if items:
                    # Try to find a heading or paragraph before the list
                    title = ""
                    prev_elem = dl.find_previous_sibling(["h1", "h2", "h3", "h4", "h5", "h6", "p"])
                    if prev_elem:
                        title = prev_elem.get_text(strip=True)
                    
                    lists.append({
                        "title": title,
                        "items": items,
                        "type": "dl",
                        "position": dl_index + 1000  # Position after standard lists
                    })
            except Exception as e:
                logger.warning(f"Error extracting definition list: {e}")
        
        return sorted(lists, key=lambda x: x.get("position", 0))
    
    def _extract_content_hierarchy(self, soup: BeautifulSoup) -> List[ContentNode]:
        """
        Extract content maintaining hierarchical structure.
        This function processes the DOM to build a tree of content nodes.
        """
        content_tree = []
        current_section = None
        current_subsection = None
        
        # Elements we're interested in for content extraction
        content_elements = [
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'p', 'div', 'span', 'article', 'section',
            'ul', 'ol', 'dl', 'table', 'blockquote',
            'pre', 'code', 'details', 'summary'
        ]
        
        # Get the main content area (article body) if available
        main_content = soup.find(['article', 'main', 'div', 'body'], 
                                class_=lambda c: c and any(x in str(c).lower() for x in 
                                                         ['content', 'article', 'main', 'body']))
        if not main_content:
            main_content = soup
        
        # Function to process a DOM node and extract meaningful content
        def extract_node_content(element, parent_node=None):
            
            # Skip invisible elements
            style = element.get('style', '')
            if 'display:none' in style or 'visibility:hidden' in style:
                # But check if it's an expandable that might be shown via JS
                if not (element.name == 'details' or 'accordion' in str(element.get('class', ''))):
                    return None
            
            # Handle different element types
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                text = element.get_text(strip=True)
                if text:
                    return ContentNode(
                        type='heading',
                        content=text,
                        level=level
                    )
            
            elif element.name == 'p':
                text = element.get_text(strip=True)
                if text:
                    return ContentNode(
                        type='paragraph',
                        content=text
                    )
            
            elif element.name in ['ul', 'ol']:
                # Only process if not already handled by extract_lists
                items = []
                for li in element.find_all('li', recursive=False):
                    item_text = li.get_text(strip=True)
                    if item_text:
                        items.append(item_text)
                
                if items:
                    # Check if there's a heading or label before this list
                    list_label = ""
                    prev_elem = element.find_previous_sibling(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'label'])
                    if prev_elem:
                        list_label = prev_elem.get_text(strip=True)
                    
                    return ContentNode(
                        type='list',
                        content=f"List: {list_label}" if list_label else "List",
                        children=[ContentNode(type='list_item', content=item) for item in items]
                    )
            
            elif element.name == 'table':
                caption = element.find('caption')
                caption_text = caption.get_text(strip=True) if caption else "Table"
                return ContentNode(
                    type='table_ref',
                    content=f"Table: {caption_text}"
                )
            
            # Handle expandable elements (details/summary, accordions)
            elif element.name == 'details':
                summary = element.find('summary')
                summary_text = summary.get_text(strip=True) if summary else "Details"
                
                details_node = ContentNode(
                    type='expandable',
                    content=f"Expandable: {summary_text}"
                )
                
                # Process contents of the expandable
                for child in element.children:
                    if isinstance(child, Tag) and child.name != 'summary':
                        child_node = extract_node_content(child, details_node)
                        if child_node:
                            details_node.children.append(child_node)
                
                return details_node
            
            # Handle divs that might contain content
            elif element.name in ['div', 'section', 'article']:
                # Check if this div directly contains text (not just from children)
                direct_text = ''.join(str(c) for c in element.contents 
                                    if isinstance(c, NavigableString)).strip()
                
                if direct_text:
                    return ContentNode(
                        type='text',
                        content=direct_text
                    )
                
                # If it has a clear role or class indicating it's a special container, create a container node
                element_class = str(element.get('class', ''))
                element_id = str(element.get('id', ''))
                element_role = str(element.get('role', ''))
                
                container_indicators = ['content', 'section', 'article', 'accordion', 'panel', 'tab']
                
                is_container = any(ind in element_class.lower() for ind in container_indicators) or \
                              any(ind in element_id.lower() for ind in container_indicators) or \
                              any(ind in element_role.lower() for ind in container_indicators)
                
                if is_container:
                    # Look for a heading inside this container
                    heading = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    heading_text = heading.get_text(strip=True) if heading else ""
                    
                    # If no heading, look for data attributes or aria labels that might indicate content
                    if not heading_text:
                        for attr in ['data-title', 'aria-label', 'title']:
                            if element.has_attr(attr):
                                heading_text = element[attr]
                                break
                    
                    container_node = ContentNode(
                        type='container',
                        content=heading_text if heading_text else "Section"
                    )
                    
                    # Process children
                    for child in element.children:
                        if isinstance(child, Tag) and child.name in content_elements:
                            child_node = extract_node_content(child, container_node)
                            if child_node:
                                container_node.children.append(child_node)
                    
                    # Only return the container if it has children
                    if container_node.children:
                        return container_node
            
            # Handle code blocks
            elif element.name in ['pre', 'code']:
                code = element.get_text()
                if code.strip():
                    return ContentNode(
                        type='code',
                        content=code
                    )
            
            # Handle blockquotes
            elif element.name == 'blockquote':
                quote = element.get_text(strip=True)
                if quote:
                    return ContentNode(
                        type='quote',
                        content=quote
                    )
            
            return None
        
        # Process all direct children of the main content area
        for element in main_content.find_all(content_elements, recursive=False):
            node = extract_node_content(element)
            if node:
                content_tree.append(node)
        
        # If we didn't find structured content, try a more aggressive approach
        if not content_tree:
            # Get all paragraphs and headings in order
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
                text = element.get_text(strip=True)
                if not text:
                    continue
                
                if element.name.startswith('h'):
                    level = int(element.name[1])
                    content_tree.append(ContentNode(
                        type='heading',
                        content=text,
                        level=level
                    ))
                elif text.strip():
                    content_tree.append(ContentNode(
                        type='paragraph',
                        content=text
                    ))
        
        # Final check - if still no content, just get all text
        if not content_tree:
            text = main_content.get_text(strip=True)
            if text:
                content_tree.append(ContentNode(
                    type='text',
                    content=text
                ))
        
        return content_tree
    
    def _get_page_content(self, url: str) -> Optional[ScrapedContent]:
        """
        Fetch and parse a single page, extracting hierarchical content.
        
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
                
                html = ""
                
                # Try Selenium first if enabled
                if self.use_selenium and self.driver:
                    try:
                        self.driver.get(url)
                        # Wait for page to load
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "body"))
                        )
                        
                        # Expand all expandable elements
                        try:
                            # Find and click on all details elements
                            details_elements = self.driver.find_elements(By.TAG_NAME, "details")
                            for element in details_elements:
                                self.driver.execute_script("arguments[0].setAttribute('open', 'true')", element)
                            
                            # Find and click on common accordion elements
                            accordion_selectors = [
                                ".accordion:not(.active)", 
                                ".collapse:not(.show)",
                                ".expandable:not(.expanded)",
                                "[data-toggle='collapse']:not(.expanded)",
                                "[aria-expanded='false']"
                            ]
                            
                            for selector in accordion_selectors:
                                try:
                                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                    for element in elements:
                                        self.driver.execute_script(
                                            "arguments[0].click(); arguments[0].setAttribute('aria-expanded', 'true');", 
                                            element
                                        )
                                except:
                                    pass
                        except Exception as e:
                            logger.warning(f"Error expanding elements: {e}")
                        
                        # Additional wait for dynamic content
                        time.sleep(2)
                        
                        html = self.driver.page_source
                    except Exception as e:
                        logger.warning(f"Selenium failed, falling back to requests: {e}")
                        self.use_selenium = False
                
                # Fallback to requests if Selenium failed or is disabled
                if not html:
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
                    
                    # Basic text for compatibility
                    text = ' '.join(article_soup.get_text().split())
                except Exception as e:
                    logger.warning(f"Readability extraction failed: {e}. Using regular HTML parsing.")
                    text = ' '.join(soup.get_text().split())
                
                # Extract structured content
                tables = self._extract_tables(soup)
                lists = self._extract_lists(soup)
                headings = defaultdict(list)
                
                # Extract headings
                for level in range(1, 7):
                    for heading in soup.find_all(f"h{level}"):
                        heading_text = heading.get_text(strip=True)
                        if heading_text:
                            headings[f"h{level}"].append(heading_text)
                
                # Extract hierarchical content structure
                content_tree = self._extract_content_hierarchy(soup)
                
                # Extract new URLs
                new_urls = self._extract_urls(soup, url)
                
                # Create scraped content object
                content = ScrapedContent(
                    url=url,
                    title=title.strip() if title else url,
                    text=text,
                    content_tree=content_tree,
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
        
        # Also save a plain text version with hierarchy preserved
        text_path = os.path.join(self.output_dir, f"{filename}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(content.to_hierarchical_text())
    
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
                        
                        # Convert content tree from dict back to ContentNode objects
                        content_tree = []
                        
                        def dict_to_content_node(node_dict):
                            node = ContentNode(
                                type=node_dict["type"],
                                content=node_dict["content"],
                                level=node_dict["level"],
                                metadata=node_dict.get("metadata", {})
                            )
                            
                            for child_dict in node_dict.get("children", []):
                                node.children.append(dict_to_content_node(child_dict))
                            
                            return node
                        
                        for node_dict in data.get("content_tree", []):
                            content_tree.append(dict_to_content_node(node_dict))
                        
                        content = ScrapedContent(
                            url=data["url"],
                            title=data["title"],
                            text=data["text"],
                            content_tree=content_tree,
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
    parser = argparse.ArgumentParser(description="Enhanced web scraper for RAG applications")
    parser.add_argument("url", help="The website URL to scrape")
    parser.add_argument("--output", "-o", default="scraped_data", help="Output directory")
    parser.add_argument("--max-pages", "-m", type=int, default=100, help="Maximum pages to scrape")
    parser.add_argument("--delay", "-d", type=float, default=1.0, help="Delay between requests in seconds")
    parser.add_argument("--threads", "-t", type=int, default=4, help="Number of threads for parallel scraping")
    parser.add_argument("--user-agent", "-u", default="RAGScraper/1.0", help="User agent string")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium for JavaScript rendering")
    parser.add_argument("--headless", action="store_true", default=True, help="Run Selenium in headless mode")
    
    args = parser.parse_args()
    
    # Create and run the scraper
    scraper = EnhancedWebScraper(
        base_url=args.url,
        output_dir=args.output,
        max_pages=args.max_pages,
        delay=args.delay,
        threads=args.threads,
        user_agent=args.user_agent,
        use_selenium=args.use_selenium,
        headless=args.headless
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
