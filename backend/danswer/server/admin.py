from typing import Any
from typing import cast

from danswer.auth.users import current_admin_user
from danswer.configs.constants import DocumentSource
from danswer.configs.constants import NO_AUTH_USER
from danswer.configs.constants import OPENAI_API_KEY_STORAGE_KEY
from danswer.connectors.factory import build_connector
from danswer.connectors.google_drive.connector_auth import get_auth_url
from danswer.connectors.google_drive.connector_auth import get_drive_tokens
from danswer.connectors.google_drive.connector_auth import save_access_tokens
from danswer.connectors.google_drive.connector_auth import verify_csrf
from danswer.connectors.models import InputType
from danswer.connectors.slack.config import get_slack_config
from danswer.connectors.slack.config import SlackConfig
from danswer.connectors.slack.config import update_slack_config
from danswer.db.index_attempt import fetch_index_attempts
from danswer.db.index_attempt import insert_index_attempt
from danswer.db.models import IndexAttempt
from danswer.db.models import IndexingStatus
from danswer.db.models import User
from danswer.direct_qa.key_validation import (
    check_openai_api_key_is_valid,
)
from danswer.direct_qa.question_answer import get_openai_api_key
from danswer.dynamic_configs import get_dynamic_config_store
from danswer.dynamic_configs.interface import ConfigNotFoundError
from danswer.server.models import ApiKey
from danswer.server.models import AuthStatus
from danswer.server.models import AuthUrl
from danswer.server.models import GDriveCallback
from danswer.server.models import IndexAttemptRequest
from danswer.server.models import IndexAttemptSnapshot
from danswer.server.models import ListIndexAttemptsResponse
from danswer.utils.logging import setup_logger
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/admin")

logger = setup_logger()


@router.get("/connectors/google-drive/check-auth", response_model=AuthStatus)
def check_drive_tokens(_: User = Depends(current_admin_user)) -> AuthStatus:
    tokens = get_drive_tokens()
    authenticated = tokens is not None
    return AuthStatus(authenticated=authenticated)


@router.get("/connectors/google-drive/authorize", response_model=AuthUrl)
def google_drive_auth(user: User = Depends(current_admin_user)) -> AuthUrl:
    user_id = str(user.id) if user else NO_AUTH_USER
    return AuthUrl(auth_url=get_auth_url(user_id))


@router.get("/connectors/google-drive/callback", status_code=201)
def google_drive_callback(
    callback: GDriveCallback = Depends(), user: User = Depends(current_admin_user)
) -> None:
    user_id = str(user.id) if user else NO_AUTH_USER
    verify_csrf(user_id, callback.state)
    return save_access_tokens(callback.code)


@router.get("/connectors/slack/config", response_model=SlackConfig)
def fetch_slack_config(_: User = Depends(current_admin_user)) -> SlackConfig:
    try:
        return get_slack_config()
    except ConfigNotFoundError:
        return SlackConfig(slack_bot_token="", workspace_id="")


@router.post("/connectors/slack/config")
def modify_slack_config(
    slack_config: SlackConfig, _: User = Depends(current_admin_user)
) -> None:
    update_slack_config(slack_config)


@router.post("/connectors/{source}/index-attempt", status_code=201)
def index(
    source: DocumentSource,
    index_attempt_request: IndexAttemptRequest,
    _: User = Depends(current_admin_user),
) -> None:
    # validate that the connector specified by the source / input_type combination
    # exists AND that the connector_specific_config is valid for that connector type, should be load
    build_connector(
        source=source,
        input_type=index_attempt_request.input_type,
        connector_specific_config=index_attempt_request.connector_specific_config,
    )

    # once validated, insert the index attempt into the database where it will
    # get picked up by a background job
    insert_index_attempt(
        index_attempt=IndexAttempt(
            source=source,
            input_type=index_attempt_request.input_type,
            connector_specific_config=index_attempt_request.connector_specific_config,
            status=IndexingStatus.NOT_STARTED,
        )
    )


@router.get("/connectors/{source}/index-attempt")
def list_index_attempts(
    source: DocumentSource,
    _: User = Depends(current_admin_user),
) -> ListIndexAttemptsResponse:
    index_attempts = fetch_index_attempts(sources=[source])
    return ListIndexAttemptsResponse(
        index_attempts=[
            IndexAttemptSnapshot(
                connector_specific_config=index_attempt.connector_specific_config,
                status=index_attempt.status,
                source=index_attempt.source,
                time_created=index_attempt.time_created,
                time_updated=index_attempt.time_updated,
                docs_indexed=0
                if not index_attempt.document_ids
                else len(index_attempt.document_ids),
            )
            for index_attempt in index_attempts
        ]
    )


@router.get("/connectors/index-attempt")
def list_all_index_attempts(
    _: User = Depends(current_admin_user),
) -> ListIndexAttemptsResponse:
    index_attempts = fetch_index_attempts()
    return ListIndexAttemptsResponse(
        index_attempts=[
            IndexAttemptSnapshot(
                connector_specific_config=index_attempt.connector_specific_config,
                status=index_attempt.status,
                source=index_attempt.source,
                time_created=index_attempt.time_created,
                time_updated=index_attempt.time_updated,
                docs_indexed=0
                if not index_attempt.document_ids
                else len(index_attempt.document_ids),
            )
            for index_attempt in index_attempts
        ]
    )


@router.head("/openai-api-key/validate")
def validate_existing_openai_api_key(
    _: User = Depends(current_admin_user),
) -> None:
    is_valid = False
    try:
        openai_api_key = get_openai_api_key()
        is_valid = check_openai_api_key_is_valid(openai_api_key)
    except ConfigNotFoundError:
        raise HTTPException(status_code=404, detail="Key not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid API key provided")


@router.get("/openai-api-key")
def get_openai_api_key_from_dynamic_config_store(
    _: User = Depends(current_admin_user),
) -> ApiKey:
    """
    NOTE: Only gets value from dynamic config store as to not expose env variables.
    """
    try:
        # only get last 4 characters of key to not expose full key
        return ApiKey(
            api_key=cast(
                str, get_dynamic_config_store().load(OPENAI_API_KEY_STORAGE_KEY)
            )[-4:]
        )
    except ConfigNotFoundError:
        raise HTTPException(status_code=404, detail="Key not found")


@router.post("/openai-api-key")
def store_openai_api_key(
    request: ApiKey,
    _: User = Depends(current_admin_user),
) -> None:
    try:
        is_valid = check_openai_api_key_is_valid(request.api_key)
        if not is_valid:
            raise HTTPException(400, "Invalid API key provided")
        get_dynamic_config_store().store(OPENAI_API_KEY_STORAGE_KEY, request.api_key)
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.delete("/openai-api-key")
def delete_openai_api_key(
    _: User = Depends(current_admin_user),
) -> None:
    get_dynamic_config_store().delete(OPENAI_API_KEY_STORAGE_KEY)
