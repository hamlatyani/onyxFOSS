import os
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.configs.constants import DocumentSource
from onyx.connectors.confluence.connector import ConfluenceConnector
from onyx.connectors.credentials_provider import OnyxStaticCredentialsProvider
from onyx.connectors.models import Document


@pytest.fixture
def confluence_connector() -> ConfluenceConnector:
    connector = ConfluenceConnector(
        wiki_base=os.environ["CONFLUENCE_TEST_SPACE_URL"],
        space=os.environ["CONFLUENCE_TEST_SPACE"],
        is_cloud=os.environ.get("CONFLUENCE_IS_CLOUD", "true").lower() == "true",
        page_id=os.environ.get("CONFLUENCE_TEST_PAGE_ID", ""),
    )

    credentials_provider = OnyxStaticCredentialsProvider(
        None,
        DocumentSource.CONFLUENCE,
        {
            "confluence_username": os.environ["CONFLUENCE_USER_NAME"],
            "confluence_access_token": os.environ["CONFLUENCE_ACCESS_TOKEN"],
        },
    )
    connector.set_credentials_provider(credentials_provider)
    return connector


@patch(
    "onyx.file_processing.extract_file_text.get_unstructured_api_key",
    return_value=None,
)
def test_confluence_connector_basic(
    mock_get_api_key: MagicMock, confluence_connector: ConfluenceConnector
) -> None:
    doc_batch_generator = confluence_connector.poll_source(0, time.time())

    doc_batch = next(doc_batch_generator)
    with pytest.raises(StopIteration):
        next(doc_batch_generator)

    assert len(doc_batch) == 2

    page_within_a_page_doc: Document | None = None
    page_doc: Document | None = None

    for doc in doc_batch:
        if doc.semantic_identifier == "DailyConnectorTestSpace Home":
            page_doc = doc
        elif doc.semantic_identifier == "Page Within A Page":
            page_within_a_page_doc = doc

    assert page_within_a_page_doc is not None
    assert page_within_a_page_doc.semantic_identifier == "Page Within A Page"
    assert page_within_a_page_doc.primary_owners
    # Updated to check for display_name instead of email
    assert page_within_a_page_doc.primary_owners[0].display_name == "Hagen O'Neill"
    assert page_within_a_page_doc.primary_owners[0].email is None
    assert len(page_within_a_page_doc.sections) == 1

    page_within_a_page_section = page_within_a_page_doc.sections[0]
    page_within_a_page_text = "@Chris Weaver loves cherry pie"
    assert page_within_a_page_section.text == page_within_a_page_text
    # Updated link assertion
    assert (
        page_within_a_page_section.link.endswith(
            "/wiki/spaces/DailyConne/pages/200769540/Page+Within+A+Page"
        )
    )

    assert page_doc is not None
    assert page_doc.semantic_identifier == "DailyConnectorTestSpace Home"
    assert page_doc.metadata["labels"] == ["testlabel"]
    assert page_doc.primary_owners
    assert page_doc.primary_owners[0].display_name == "Hagen O'Neill"
    assert page_doc.primary_owners[0].email is None
    assert len(page_doc.sections) == 2

    page_section = page_doc.sections[0]
    assert page_section.text == "test123 " + page_within_a_page_text
    assert page_section.link.endswith("/wiki/spaces/DailyConne/overview")