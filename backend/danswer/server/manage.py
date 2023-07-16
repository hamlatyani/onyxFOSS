from datetime import datetime
from datetime import timedelta
from typing import cast

from danswer.auth.schemas import UserRole
from danswer.auth.users import current_admin_user
from danswer.auth.users import current_user
from danswer.configs.app_configs import DISABLE_GENERATIVE_AI
from danswer.configs.app_configs import GENERATIVE_MODEL_ACCESS_CHECK_FREQ
from danswer.configs.app_configs import MASK_CREDENTIAL_PREFIX
from danswer.configs.constants import OPENAI_API_KEY_STORAGE_KEY
from danswer.connectors.file.utils import write_temp_files
from danswer.connectors.google_drive.connector_auth import DB_CREDENTIALS_DICT_KEY
from danswer.connectors.google_drive.connector_auth import get_auth_url
from danswer.connectors.google_drive.connector_auth import get_drive_tokens
from danswer.connectors.google_drive.connector_auth import get_google_app_cred
from danswer.connectors.google_drive.connector_auth import (
    update_credential_access_tokens,
)
from danswer.connectors.google_drive.connector_auth import upsert_google_app_cred
from danswer.connectors.google_drive.connector_auth import verify_csrf
from danswer.db.connector import create_connector
from danswer.db.connector import delete_connector
from danswer.db.connector import fetch_connector_by_id
from danswer.db.connector import fetch_connectors
from danswer.db.connector import get_connector_credential_ids
from danswer.db.connector import update_connector
from danswer.db.connector_credential_pair import add_credential_to_connector
from danswer.db.connector_credential_pair import get_connector_credential_pairs
from danswer.db.connector_credential_pair import remove_credential_from_connector
from danswer.db.credentials import create_credential
from danswer.db.credentials import delete_credential
from danswer.db.credentials import fetch_credential_by_id
from danswer.db.credentials import fetch_credentials
from danswer.db.credentials import mask_credential_dict
from danswer.db.credentials import update_credential
from danswer.db.engine import get_session
from danswer.db.engine import get_sqlalchemy_async_engine
from danswer.db.index_attempt import create_index_attempt
from danswer.db.models import User
from danswer.direct_qa.key_validation import check_openai_api_key_is_valid
from danswer.direct_qa.llm import get_openai_api_key
from danswer.dynamic_configs import get_dynamic_config_store
from danswer.dynamic_configs.interface import ConfigNotFoundError
from danswer.server.models import ApiKey
from danswer.server.models import AuthStatus
from danswer.server.models import AuthUrl
from danswer.server.models import ConnectorBase
from danswer.server.models import ConnectorIndexingStatus
from danswer.server.models import ConnectorSnapshot
from danswer.server.models import CredentialBase
from danswer.server.models import CredentialSnapshot
from danswer.server.models import FileUploadResponse
from danswer.server.models import GDriveCallback
from danswer.server.models import GoogleAppCredentials
from danswer.server.models import ObjectCreationIdResponse
from danswer.server.models import RunConnectorRequest
from danswer.server.models import StatusResponse
from danswer.server.models import UserByEmail
from danswer.server.models import UserRoleResponse
from danswer.utils.logger import setup_logger
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Request
from fastapi import Response
from fastapi import UploadFile
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session


router = APIRouter(prefix="/manage")
logger = setup_logger()

_GOOGLE_DRIVE_CREDENTIAL_ID_COOKIE_NAME = "google_drive_credential_id"


"""Admin only API endpoints"""


@router.patch("/promote-user-to-admin", response_model=None)
async def promote_admin(
    user_email: UserByEmail, user: User = Depends(current_admin_user)
) -> None:
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    async with AsyncSession(get_sqlalchemy_async_engine()) as asession:
        user_db = SQLAlchemyUserDatabase(asession, User)  # type: ignore
        user_to_promote = await user_db.get_by_email(user_email.user_email)
        if not user_to_promote:
            raise HTTPException(status_code=404, detail="User not found")
        user_to_promote.role = UserRole.ADMIN
        asession.add(user_to_promote)
        await asession.commit()
    return


@router.get("/admin/connector/google-drive/app-credential")
def check_google_app_credentials_exist(
    _: User = Depends(current_admin_user),
) -> dict[str, str]:
    try:
        return {"client_id": get_google_app_cred().web.client_id}
    except ConfigNotFoundError as e:
        raise HTTPException(status_code=404, detail="Google App Credentials not found")


@router.put("/admin/connector/google-drive/app-credential")
def update_google_app_credentials(
    app_credentials: GoogleAppCredentials, _: User = Depends(current_admin_user)
) -> StatusResponse:
    try:
        upsert_google_app_cred(app_credentials)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StatusResponse(
        success=True, message="Successfully saved Google App Credentials"
    )


@router.get("/admin/connector/google-drive/check-auth/{credential_id}")
def check_drive_tokens(
    credential_id: int,
    user: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> AuthStatus:
    db_credentials = fetch_credential_by_id(credential_id, user, db_session)
    if (
        not db_credentials
        or DB_CREDENTIALS_DICT_KEY not in db_credentials.credential_json
    ):
        return AuthStatus(authenticated=False)
    token_json_str = str(db_credentials.credential_json[DB_CREDENTIALS_DICT_KEY])
    google_drive_creds = get_drive_tokens(token_json_str=token_json_str)
    if google_drive_creds is None:
        return AuthStatus(authenticated=False)
    return AuthStatus(authenticated=True)


@router.get("/admin/connector/google-drive/authorize/{credential_id}")
def admin_google_drive_auth(
    response: Response, credential_id: str, _: User = Depends(current_admin_user)
) -> AuthUrl:
    # set a cookie that we can read in the callback (used for `verify_csrf`)
    response.set_cookie(
        key=_GOOGLE_DRIVE_CREDENTIAL_ID_COOKIE_NAME,
        value=credential_id,
        httponly=True,
        max_age=600,
    )
    return AuthUrl(auth_url=get_auth_url(credential_id=int(credential_id)))


@router.post("/admin/connector/file/upload")
def upload_files(
    files: list[UploadFile], _: User = Depends(current_admin_user)
) -> FileUploadResponse:
    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="File name cannot be empty")
    try:
        file_paths = write_temp_files(
            [(cast(str, file.filename), file.file) for file in files]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FileUploadResponse(file_paths=file_paths)


@router.get("/admin/connector/indexing-status")
def get_connector_indexing_status(
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> list[ConnectorIndexingStatus]:
    indexing_statuses: list[ConnectorIndexingStatus] = []

    cc_pairs = get_connector_credential_pairs(db_session)
    for cc_pair in cc_pairs:
        connector = cc_pair.connector
        credential = cc_pair.credential
        indexing_statuses.append(
            ConnectorIndexingStatus(
                connector=ConnectorSnapshot.from_connector_db_model(connector),
                public_doc=credential.public_doc,
                owner=credential.user.email if credential.user else "",
                last_status=cc_pair.last_attempt_status,
                last_success=cc_pair.last_successful_index_time,
                docs_indexed=cc_pair.total_docs_indexed,
            )
        )

    return indexing_statuses


@router.post("/admin/connector")
def create_connector_from_model(
    connector_info: ConnectorBase,
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> ObjectCreationIdResponse:
    try:
        return create_connector(connector_info, db_session)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.patch("/admin/connector/{connector_id}")
def update_connector_from_model(
    connector_id: int,
    connector_data: ConnectorBase,
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> ConnectorSnapshot | StatusResponse[int]:
    updated_connector = update_connector(connector_id, connector_data, db_session)
    if updated_connector is None:
        raise HTTPException(
            status_code=404, detail=f"Connector {connector_id} does not exist"
        )

    return ConnectorSnapshot(
        id=updated_connector.id,
        name=updated_connector.name,
        source=updated_connector.source,
        input_type=updated_connector.input_type,
        connector_specific_config=updated_connector.connector_specific_config,
        refresh_freq=updated_connector.refresh_freq,
        credential_ids=[
            association.credential.id for association in updated_connector.credentials
        ],
        time_created=updated_connector.time_created,
        time_updated=updated_connector.time_updated,
        disabled=updated_connector.disabled,
    )


@router.delete("/admin/connector/{connector_id}", response_model=StatusResponse[int])
def delete_connector_by_id(
    connector_id: int,
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse[int]:
    return delete_connector(connector_id, db_session)


@router.post("/admin/connector/run-once")
def connector_run_once(
    run_info: RunConnectorRequest,
    _: User = Depends(current_admin_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse[list[int]]:
    connector_id = run_info.connector_id
    specified_credential_ids = run_info.credential_ids
    try:
        possible_credential_ids = get_connector_credential_ids(
            run_info.connector_id, db_session
        )
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Connector by id {connector_id} does not exist.",
        )

    if not specified_credential_ids:
        credential_ids = possible_credential_ids
    else:
        if set(specified_credential_ids).issubset(set(possible_credential_ids)):
            credential_ids = specified_credential_ids
        else:
            raise HTTPException(
                status_code=400,
                detail="Not all specified credentials are associated with connector",
            )

    if not credential_ids:
        raise HTTPException(
            status_code=400,
            detail="Connector has no valid credentials, cannot create index attempts.",
        )

    index_attempt_ids = [
        create_index_attempt(run_info.connector_id, credential_id, db_session)
        for credential_id in credential_ids
    ]
    return StatusResponse(
        success=True,
        message=f"Successfully created {len(index_attempt_ids)} index attempts",
        data=index_attempt_ids,
    )


@router.head("/admin/openai-api-key/validate")
def validate_existing_openai_api_key(
    _: User = Depends(current_admin_user),
) -> None:
    # OpenAI key is only used for generative QA, so no need to validate this
    # if it's turned off
    if DISABLE_GENERATIVE_AI:
        return

    # always check if key exists
    try:
        openai_api_key = get_openai_api_key()
    except ConfigNotFoundError:
        raise HTTPException(status_code=404, detail="Key not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # don't call OpenAI every single time, only validate every so often
    check_key_time = "openai_api_key_last_check_time"
    kv_store = get_dynamic_config_store()
    curr_time = datetime.now()
    try:
        last_check = datetime.fromtimestamp(cast(float, kv_store.load(check_key_time)))
        check_freq_sec = timedelta(seconds=GENERATIVE_MODEL_ACCESS_CHECK_FREQ)
        if curr_time - last_check < check_freq_sec:
            return
    except ConfigNotFoundError:
        # First time checking the key, nothing unusual
        pass

    get_dynamic_config_store().store(check_key_time, curr_time.timestamp())

    try:
        is_valid = check_openai_api_key_is_valid(openai_api_key)
    except ValueError:
        # this is the case where they aren't using an OpenAI-based model
        is_valid = True

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid API key provided")


@router.get("/admin/openai-api-key", response_model=ApiKey)
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


@router.put("/admin/openai-api-key")
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


@router.delete("/admin/openai-api-key")
def delete_openai_api_key(
    _: User = Depends(current_admin_user),
) -> None:
    get_dynamic_config_store().delete(OPENAI_API_KEY_STORAGE_KEY)


"""Endpoints for basic users"""


@router.get("/get-user-role", response_model=UserRoleResponse)
async def get_user_role(user: User = Depends(current_user)) -> UserRoleResponse:
    if user is None:
        raise ValueError("Invalid or missing user.")
    return UserRoleResponse(role=user.role)


@router.get("/connector/google-drive/authorize/{credential_id}")
def google_drive_auth(
    response: Response, credential_id: str, _: User = Depends(current_user)
) -> AuthUrl:
    # set a cookie that we can read in the callback (used for `verify_csrf`)
    response.set_cookie(
        key=_GOOGLE_DRIVE_CREDENTIAL_ID_COOKIE_NAME,
        value=credential_id,
        httponly=True,
        max_age=600,
    )
    return AuthUrl(auth_url=get_auth_url(int(credential_id)))


@router.get("/connector/google-drive/callback")
def google_drive_callback(
    request: Request,
    callback: GDriveCallback = Depends(),
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    credential_id_cookie = request.cookies.get(_GOOGLE_DRIVE_CREDENTIAL_ID_COOKIE_NAME)
    if credential_id_cookie is None or not credential_id_cookie.isdigit():
        raise HTTPException(
            status_code=401, detail="Request did not pass CSRF verification."
        )
    credential_id = int(credential_id_cookie)
    verify_csrf(credential_id, callback.state)
    if (
        update_credential_access_tokens(callback.code, credential_id, user, db_session)
        is None
    ):
        raise HTTPException(
            status_code=500, detail="Unable to fetch Google Drive access tokens"
        )

    return StatusResponse(success=True, message="Updated Google Drive access tokens")


@router.get("/connector")
def get_connectors(
    _: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[ConnectorSnapshot]:
    connectors = fetch_connectors(db_session)
    return [
        ConnectorSnapshot.from_connector_db_model(connector) for connector in connectors
    ]


@router.get("/connector/{connector_id}")
def get_connector_by_id(
    connector_id: int,
    _: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ConnectorSnapshot | StatusResponse[int]:
    connector = fetch_connector_by_id(connector_id, db_session)
    if connector is None:
        raise HTTPException(
            status_code=404, detail=f"Connector {connector_id} does not exist"
        )

    return ConnectorSnapshot(
        id=connector.id,
        name=connector.name,
        source=connector.source,
        input_type=connector.input_type,
        connector_specific_config=connector.connector_specific_config,
        refresh_freq=connector.refresh_freq,
        credential_ids=[
            association.credential.id for association in connector.credentials
        ],
        time_created=connector.time_created,
        time_updated=connector.time_updated,
        disabled=connector.disabled,
    )


@router.get("/credential")
def get_credentials(
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> list[CredentialSnapshot]:
    credentials = fetch_credentials(user, db_session)
    return [
        CredentialSnapshot(
            id=credential.id,
            credential_json=mask_credential_dict(credential.credential_json)
            if MASK_CREDENTIAL_PREFIX
            else credential.credential_json,
            user_id=credential.user_id,
            public_doc=credential.public_doc,
            time_created=credential.time_created,
            time_updated=credential.time_updated,
        )
        for credential in credentials
    ]


@router.get("/credential/{credential_id}")
def get_credential_by_id(
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CredentialSnapshot | StatusResponse[int]:
    credential = fetch_credential_by_id(credential_id, user, db_session)
    if credential is None:
        raise HTTPException(
            status_code=401,
            detail=f"Credential {credential_id} does not exist or does not belong to user",
        )

    return CredentialSnapshot(
        id=credential.id,
        credential_json=mask_credential_dict(credential.credential_json)
        if MASK_CREDENTIAL_PREFIX
        else credential.credential_json,
        user_id=credential.user_id,
        public_doc=credential.public_doc,
        time_created=credential.time_created,
        time_updated=credential.time_updated,
    )


@router.post("/credential")
def create_credential_from_model(
    connector_info: CredentialBase,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> ObjectCreationIdResponse:
    return create_credential(connector_info, user, db_session)


@router.patch("/credential/{credential_id}")
def update_credential_from_model(
    credential_id: int,
    credential_data: CredentialBase,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> CredentialSnapshot | StatusResponse[int]:
    updated_credential = update_credential(
        credential_id, credential_data, user, db_session
    )
    if updated_credential is None:
        raise HTTPException(
            status_code=401,
            detail=f"Credential {credential_id} does not exist or does not belong to user",
        )

    return CredentialSnapshot(
        id=updated_credential.id,
        credential_json=updated_credential.credential_json,
        user_id=updated_credential.user_id,
        public_doc=updated_credential.public_doc,
        time_created=updated_credential.time_created,
        time_updated=updated_credential.time_updated,
    )


@router.delete("/credential/{credential_id}")
def delete_credential_by_id(
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse:
    delete_credential(credential_id, user, db_session)
    return StatusResponse(
        success=True, message="Credential deleted successfully", data=credential_id
    )


@router.put("/connector/{connector_id}/credential/{credential_id}")
def associate_credential_to_connector(
    connector_id: int,
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse[int]:
    return add_credential_to_connector(connector_id, credential_id, user, db_session)


@router.delete("/connector/{connector_id}/credential/{credential_id}")
def dissociate_credential_from_connector(
    connector_id: int,
    credential_id: int,
    user: User = Depends(current_user),
    db_session: Session = Depends(get_session),
) -> StatusResponse[int]:
    return remove_credential_from_connector(
        connector_id, credential_id, user, db_session
    )
