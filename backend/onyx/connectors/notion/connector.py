from collections.abc import Generator
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import cast
from typing import Optional

import requests
from pydantic import BaseModel
from retry import retry

# Assurez-vous que ces imports correspondent à votre environnement
from onyx.configs.app_configs import INDEX_BATCH_SIZE
from onyx.configs.app_configs import NOTION_CONNECTOR_DISABLE_RECURSIVE_PAGE_LOOKUP
from onyx.configs.constants import DocumentSource
from onyx.connectors.cross_connector_utils.rate_limit_wrapper import (
    rl_requests,
)
from onyx.connectors.exceptions import ConnectorValidationError
from onyx.connectors.exceptions import CredentialExpiredError
from onyx.connectors.exceptions import InsufficientPermissionsError
from onyx.connectors.exceptions import UnexpectedValidationError
from onyx.connectors.interfaces import GenerateDocumentsOutput
from onyx.connectors.interfaces import LoadConnector
from onyx.connectors.interfaces import PollConnector
from onyx.connectors.interfaces import SecondsSinceUnixEpoch
from onyx.connectors.models import ConnectorMissingCredentialError
from onyx.connectors.models import Document
from onyx.connectors.models import ImageSection
from onyx.connectors.models import TextSection
from onyx.utils.batching import batch_generator
from onyx.utils.logger import setup_logger

logger = setup_logger()

_NOTION_PAGE_SIZE = 100
_NOTION_CALL_TIMEOUT = 30  # 30 seconds


class NotionPage(BaseModel):
    """Represents a Notion Page object"""
    id: str
    created_time: str
    last_edited_time: str
    archived: bool
    properties: dict[str, Any]
    url: str
    database_name: str | None = None


class NotionBlock(BaseModel):
    """Represents a Notion Block object"""
    id: str
    text: str
    prefix: str


class NotionSearchResponse(BaseModel):
    """Represents the response from the Notion Search API"""
    results: list[dict[str, Any]]
    next_cursor: Optional[str]
    has_more: bool = False


class NotionConnector(LoadConnector, PollConnector):
    def __init__(
            self,
            batch_size: int = INDEX_BATCH_SIZE,
            recursive_index_enabled: bool = not NOTION_CONNECTOR_DISABLE_RECURSIVE_PAGE_LOOKUP,
            root_page_id: str | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.headers = {
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",  # Version compatible avec la logique d'extraction
        }
        self.indexed_pages: set[str] = set()
        self.root_page_id = root_page_id
        self.recursive_index_enabled = recursive_index_enabled or self.root_page_id

    @retry(tries=3, delay=1, backoff=2)
    def _fetch_child_blocks(
            self, block_id: str, cursor: str | None = None
    ) -> dict[str, Any] | None:
        logger.debug(f"Fetching children of block with ID '{block_id}'")
        block_url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        query_params = None if not cursor else {"start_cursor": cursor}
        res = rl_requests.get(
            block_url,
            headers=self.headers,
            params=query_params,
            timeout=_NOTION_CALL_TIMEOUT,
        )
        try:
            res.raise_for_status()
        except Exception as e:
            if res.status_code == 404:
                logger.error(
                    f"Unable to access block with ID '{block_id}'. "
                    f"Likely not shared with integration. Exception:\n\n{e}"
                )
            else:
                logger.exception(
                    f"Error fetching blocks with status code {res.status_code}: {res.json()}"
                )
            return None
        return res.json()

    @retry(tries=3, delay=1, backoff=2)
    def _fetch_page(self, page_id: str) -> NotionPage:
        logger.debug(f"Fetching page for ID '{page_id}'")
        page_url = f"https://api.notion.com/v1/pages/{page_id}"
        res = rl_requests.get(
            page_url,
            headers=self.headers,
            timeout=_NOTION_CALL_TIMEOUT,
        )
        try:
            res.raise_for_status()
        except Exception as e:
            logger.warning(
                f"Failed to fetch page, trying database for ID '{page_id}'. Exception: {e}"
            )
            return self._fetch_database_as_page(page_id)
        return NotionPage(**res.json())

    @retry(tries=3, delay=1, backoff=2)
    def _fetch_database_as_page(self, database_id: str) -> NotionPage:
        logger.debug(f"Fetching database for ID '{database_id}' as a page")
        database_url = f"https://api.notion.com/v1/databases/{database_id}"
        res = rl_requests.get(
            database_url,
            headers=self.headers,
            timeout=_NOTION_CALL_TIMEOUT,
        )
        try:
            res.raise_for_status()
        except Exception as e:
            logger.exception(f"Error fetching database as page - {res.json()}")
            raise e
        database_name_list = res.json().get("title", [])
        database_name = (
            database_name_list[0].get("text", {}).get("content") if database_name_list else None
        )
        return NotionPage(**res.json(), database_name=database_name)

    @retry(tries=3, delay=1, backoff=2)
    def _search_notion(self, query: dict[str, Any]) -> NotionSearchResponse:
        logger.debug(f"Searching Notion with query: {query}")
        try:
            res = rl_requests.post(
                "https://api.notion.com/v1/search",
                headers=self.headers,
                json=query,
                timeout=_NOTION_CALL_TIMEOUT,
            )
            res.raise_for_status()
        except Exception as e:
            logger.exception(f"Error searching Notion: {e}")
            raise e
        return NotionSearchResponse(**res.json())

    @staticmethod
    def _properties_to_str(properties: dict[str, Any]) -> str:
        """Converts Notion properties to a string using recursive logic."""
        def _recurse_list_properties(inner_list: list[Any]) -> str | None:
            list_properties: list[str | None] = []
            for item in inner_list:
                if item and isinstance(item, dict):
                    list_properties.append(_recurse_properties(item))
                elif item and isinstance(item, list):
                    list_properties.append(_recurse_list_properties(item))
                else:
                    list_properties.append(str(item))
            return (", ".join([p for p in list_properties if p]) or None)

        def _recurse_properties(inner_dict: dict[str, Any]) -> str | None:
            sub_inner_dict: Any = inner_dict
            while isinstance(sub_inner_dict, dict) and "type" in sub_inner_dict:
                type_name = sub_inner_dict["type"]
                sub_inner_dict = sub_inner_dict[type_name]
                if not sub_inner_dict:
                    return None

            if isinstance(sub_inner_dict, list):
                return _recurse_list_properties(sub_inner_dict)
            elif isinstance(sub_inner_dict, str):
                return sub_inner_dict
            elif isinstance(sub_inner_dict, dict):
                if "name" in sub_inner_dict:
                    return sub_inner_dict["name"]
                if "content" in sub_inner_dict:
                    return sub_inner_dict["content"]
                start = sub_inner_dict.get("start")
                end = sub_inner_dict.get("end")
                if start:
                    return f"{start} - {end}" if end else start
                elif end:
                    return f"Until {end}"
                if "id" in sub_inner_dict:
                    return None
            logger.debug(f"Unreadable property from innermost prop: {sub_inner_dict}")
            return None

        result = []
        for prop_name, prop in properties.items():
            if not prop or not isinstance(prop, dict):
                continue
            try:
                inner_value = _recurse_properties(prop)
                if inner_value:
                    result.append(f"{prop_name}: {inner_value}")
            except Exception as e:
                logger.warning(f"Error recursing properties for {prop_name}: {e}")
                continue
        return "\n".join(result)

    def _read_blocks(self, base_block_id: str) -> tuple[list[NotionBlock], list[str]]:
        """Reads all child blocks for the specified block, returns a list of blocks and child page ids"""
        result_blocks: list[NotionBlock] = []
        child_pages: list[str] = []
        cursor = None
        while True:
            data = self._fetch_child_blocks(base_block_id, cursor)
            if data is None:
                return result_blocks, child_pages

            for result in data["results"]:
                result_id = result["id"]
                result_type = result["type"]
                result_obj = result[result_type]

                cur_result_text_arr = []
                if "rich_text" in result_obj:
                    for rich_text in result_obj["rich_text"]:
                        if "text" in rich_text:
                            cur_result_text_arr.append(rich_text["text"]["content"])

                if result.get("has_children"):
                    if result_type == "child_page":
                        child_pages.append(result_id)
                    else:
                        subblocks, subblock_child_pages = self._read_blocks(result_id)
                        result_blocks.extend(subblocks)
                        child_pages.extend(subblock_child_pages)

                if cur_result_text_arr:
                    new_block = NotionBlock(
                        id=result_id,
                        text="\n".join(cur_result_text_arr),
                        prefix="\n",
                    )
                    result_blocks.append(new_block)

            if not data.get("has_more") or not data.get("next_cursor"):
                break
            cursor = data["next_cursor"]
        return result_blocks, child_pages

    def _read_page_title(self, page: NotionPage) -> str:
        if page.database_name:
            return page.database_name
        for _, prop in page.properties.items():
            if prop.get("type") == "title" and prop.get("title"):
                return " ".join([t.get("plain_text", "") for t in prop["title"]]).strip()
        return f"Untitled Page {page.id}"

    def _read_pages(self, pages: list[NotionPage]) -> Generator[Document, None, None]:
        """Reads pages and generates Documents based on sections, optimized for RAG."""
        for page in pages:
            if page.id in self.indexed_pages:
                continue
            self.indexed_pages.add(page.id)

            logger.info(f"Reading page with ID '{page.id}', with url {page.url}")
            title = self._read_page_title(page)
            page_blocks, child_page_ids = self._read_blocks(page.id)

            sections = []

            properties_text = self._properties_to_str(page.properties)
            if properties_text:
                sections.append(TextSection(text=properties_text, link=page.url))

            for block in page_blocks:
                block_link = f"{page.url}#{block.id.replace('-', '')}"
                sections.append(TextSection(text=block.prefix + block.text, link=block_link))

            if not sections:
                logger.warning(f"Page with ID '{page.id}' has no content. Skipping.")
                continue

            created_at = datetime.fromisoformat(page.created_time).replace(tzinfo=timezone.utc)
            last_edited_at = datetime.fromisoformat(page.last_edited_time).replace(tzinfo=timezone.utc)

            # Création du document SANS le champ 'content' global pour forcer l'utilisation des sections
            yield Document(
                source=DocumentSource.NOTION,
                id=page.id,
                semantic_identifier=title,
                title=title,
                doc_created_at=created_at,
                doc_updated_at=last_edited_at,
                metadata={
                    "source_url": page.url,
                    "created_at": created_at.isoformat(),
                    "last_edited_at": last_edited_at.isoformat(),
                },
                sections=cast(list[TextSection | ImageSection], sections),
            )

            if self.recursive_index_enabled and child_page_ids:
                pages_to_fetch = [pid for pid in child_page_ids if pid not in self.indexed_pages]
                if pages_to_fetch:
                    child_page_objects = [self._fetch_page(page_id) for page_id in pages_to_fetch]
                    yield from self._read_pages(child_page_objects)

    def _filter_pages_by_time(
            self,
            pages: list[dict[str, Any]],
            start: SecondsSinceUnixEpoch,
            end: SecondsSinceUnixEpoch,
            filter_field: str = "last_edited_time",
    ) -> list[NotionPage]:
        filtered_pages: list[NotionPage] = []
        for page in pages:
            timestamp_str = page[filter_field].replace(".000Z", "+00:00")
            compare_time = datetime.fromisoformat(timestamp_str).timestamp()
            if start < compare_time <= end:
                filtered_pages.append(NotionPage(**page))
        return filtered_pages

    def _recursive_load(self) -> Generator[list[Document], None, None]:
        if self.root_page_id is None or not self.recursive_index_enabled:
            raise RuntimeError("Recursive page lookup is not enabled, but trying to recursively load pages.")

        logger.info(f"Recursively loading pages from Notion based on root page with ID: {self.root_page_id}")
        pages = [self._fetch_page(page_id=self.root_page_id)]
        yield from batch_generator(self._read_pages(pages), self.batch_size)

    def load_credentials(self, credentials: dict[str, Any]) -> dict[str, Any] | None:
        self.headers["Authorization"] = f'Bearer {credentials["notion_integration_token"]}'
        return None

    def load_from_state(self) -> GenerateDocumentsOutput:
        if self.recursive_index_enabled and self.root_page_id:
            yield from self._recursive_load()
            return

        query_dict = {
            "filter": {"property": "object", "value": "page"},
            "page_size": _NOTION_PAGE_SIZE,
        }
        while True:
            db_res = self._search_notion(query_dict)
            pages = [NotionPage(**page) for page in db_res.results]
            yield from batch_generator(self._read_pages(pages), self.batch_size)
            if db_res.has_more and db_res.next_cursor:
                query_dict["start_cursor"] = db_res.next_cursor
            else:
                break

    def poll_source(
            self, start: SecondsSinceUnixEpoch, end: SecondsSinceUnixEpoch
    ) -> GenerateDocumentsOutput:
        if self.recursive_index_enabled and self.root_page_id:
            yield from self._recursive_load()
            return

        query_dict = {
            "page_size": _NOTION_PAGE_SIZE,
            "sort": {"timestamp": "last_edited_time", "direction": "descending"},
            "filter": {"property": "object", "value": "page"},
        }
        while True:
            db_res = self._search_notion(query_dict)
            pages = self._filter_pages_by_time(
                db_res.results, start, end, filter_field="last_edited_time"
            )
            if pages:
                yield from batch_generator(self._read_pages(pages), self.batch_size)
                if db_res.has_more and db_res.next_cursor:
                    query_dict["start_cursor"] = db_res.next_cursor
                else:
                    break
            else:
                break

    def validate_connector_settings(self) -> None:
        if not self.headers.get("Authorization"):
            raise ConnectorMissingCredentialError("Notion credentials not loaded.")

        try:
            if self.root_page_id:
                res = rl_requests.get(
                    f"https://api.notion.com/v1/pages/{self.root_page_id}",
                    headers=self.headers,
                    timeout=_NOTION_CALL_TIMEOUT,
                )
            else:
                test_query = {
                    "filter": {"property": "object", "value": "page"},
                    "page_size": 1,
                }
                res = rl_requests.post(
                    "https://api.notion.com/v1/search",
                    headers=self.headers,
                    json=test_query,
                    timeout=_NOTION_CALL_TIMEOUT,
                )
            res.raise_for_status()

        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code if http_err.response else None
            if status_code == 401:
                raise CredentialExpiredError("Notion credential appears to be invalid or expired (HTTP 401).")
            elif status_code == 403:
                raise InsufficientPermissionsError("Your Notion token does not have sufficient permissions (HTTP 403).")
            elif status_code == 404:
                raise ConnectorValidationError("Notion resource not found or not shared with the integration (HTTP 404).")
            elif status_code == 429:
                raise ConnectorValidationError("Validation failed due to Notion rate-limits being exceeded (HTTP 429). Please try again later.")
            else:
                raise UnexpectedValidationError(f"Unexpected Notion HTTP error (status={status_code}): {http_err}") from http_err
        except Exception as exc:
            raise UnexpectedValidationError(f"Unexpected error during Notion settings validation: {exc}")


if __name__ == "__main__":
    import os

    root_page_id = os.environ.get("NOTION_ROOT_PAGE_ID")
    connector = NotionConnector(root_page_id=root_page_id)
    connector.load_credentials(
        {"notion_integration_token": os.environ.get("NOTION_INTEGRATION_TOKEN")}
    )
    document_batches = connector.load_from_state()
    for doc_batch in document_batches:
        for doc in doc_batch:
            print(doc)