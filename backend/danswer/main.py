import nltk  # type:ignore
import uvicorn
from danswer.auth.schemas import UserCreate
from danswer.auth.schemas import UserRead
from danswer.auth.schemas import UserUpdate
from danswer.auth.users import auth_backend
from danswer.auth.users import fastapi_users
from danswer.auth.users import google_oauth_client
from danswer.configs.app_configs import APP_HOST
from danswer.configs.app_configs import APP_PORT
from danswer.configs.app_configs import DISABLE_AUTH
from danswer.configs.app_configs import ENABLE_OAUTH
from danswer.configs.app_configs import GOOGLE_OAUTH_CLIENT_ID
from danswer.configs.app_configs import GOOGLE_OAUTH_CLIENT_SECRET
from danswer.configs.app_configs import SECRET
from danswer.configs.app_configs import TYPESENSE_DEFAULT_COLLECTION
from danswer.configs.app_configs import WEB_DOMAIN
from danswer.datastores.qdrant.indexing import list_qdrant_collections
from danswer.datastores.typesense.store import check_typesense_collection_exist
from danswer.datastores.typesense.store import create_typesense_collection
from danswer.db.credentials import create_initial_public_credential
from danswer.direct_qa.key_validation import check_openai_api_key_is_valid
from danswer.direct_qa.llm import get_openai_api_key
from danswer.dynamic_configs.interface import ConfigNotFoundError
from danswer.server.event_loading import router as event_processing_router
from danswer.server.health import router as health_router
from danswer.server.manage import router as admin_router
from danswer.server.search_backend import router as backend_router
from danswer.utils.logger import setup_logger
from fastapi import FastAPI
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


logger = setup_logger()


def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logger.exception(f"{request}: {exc_str}")
    content = {"status_code": 422, "message": exc_str, "data": None}
    return JSONResponse(content=content, status_code=422)


def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


def get_application() -> FastAPI:
    application = FastAPI(title="Internal Search QA Backend", debug=True, version="0.1")
    application.include_router(backend_router)
    application.include_router(event_processing_router)
    application.include_router(admin_router)
    application.include_router(health_router)

    application.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/auth/database",
        tags=["auth"],
    )
    application.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
        tags=["auth"],
    )
    application.include_router(
        fastapi_users.get_reset_password_router(),
        prefix="/auth",
        tags=["auth"],
    )
    application.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/auth",
        tags=["auth"],
    )
    application.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate),
        prefix="/users",
        tags=["users"],
    )
    if ENABLE_OAUTH:
        application.include_router(
            fastapi_users.get_oauth_router(
                google_oauth_client,
                auth_backend,
                SECRET,
                associate_by_email=True,
                is_verified_by_default=True,
                # points the user back to the login page, where we will call the
                # /auth/google/callback endpoint + redirect them to the main app
                redirect_url=f"{WEB_DOMAIN}/auth/google/callback",
            ),
            prefix="/auth/google",
            tags=["auth"],
        )
        application.include_router(
            fastapi_users.get_oauth_associate_router(
                google_oauth_client, UserRead, SECRET
            ),
            prefix="/auth/associate/google",
            tags=["auth"],
        )

    application.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )

    application.add_exception_handler(ValueError, value_error_handler)

    @application.on_event("startup")
    def startup_event() -> None:
        # To avoid circular imports
        from danswer.search.search_utils import (
            warm_up_models,
        )
        from danswer.datastores.qdrant.indexing import create_qdrant_collection
        from danswer.configs.app_configs import QDRANT_DEFAULT_COLLECTION

        auth_status = "off" if DISABLE_AUTH else "on"
        logger.info(f"User auth is turned {auth_status}")

        if not ENABLE_OAUTH:
            logger.debug("OAuth is turned off")
        else:
            if not GOOGLE_OAUTH_CLIENT_ID or not GOOGLE_OAUTH_CLIENT_SECRET:
                logger.warning("OAuth is turned on but incorrectly configured")
            else:
                logger.debug("OAuth is turned on")

        logger.info("Warming up local NLP models.")
        warm_up_models()

        logger.info("Verifying query preprocessing (NLTK) data is downloaded")
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("punkt")

        logger.info("Verifying public credential exists.")
        create_initial_public_credential()

        logger.info("Verifying Document Indexes are available.")
        if QDRANT_DEFAULT_COLLECTION not in {
            collection.name for collection in list_qdrant_collections().collections
        }:
            logger.info(
                f"Creating Qdrant collection with name: {QDRANT_DEFAULT_COLLECTION}"
            )
            create_qdrant_collection(collection_name=QDRANT_DEFAULT_COLLECTION)
        if not check_typesense_collection_exist(TYPESENSE_DEFAULT_COLLECTION):
            logger.info(
                f"Creating Typesense collection with name: {TYPESENSE_DEFAULT_COLLECTION}"
            )
            create_typesense_collection(collection_name=TYPESENSE_DEFAULT_COLLECTION)

    return application


app = get_application()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the list of allowed origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    logger.info(f"Running QA Service on http://{APP_HOST}:{str(APP_PORT)}/")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
