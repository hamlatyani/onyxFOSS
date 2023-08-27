import json
from collections.abc import Generator
from dataclasses import asdict

from fastapi import APIRouter
from fastapi import Depends
from fastapi.responses import StreamingResponse

from danswer.auth.users import current_user
from danswer.chunking.models import InferenceChunk
from danswer.configs.app_configs import DISABLE_GENERATIVE_AI
from danswer.configs.app_configs import NUM_GENERATIVE_AI_INPUT_DOCS
from danswer.datastores.document_index import get_default_document_index
from danswer.db.models import User
from danswer.direct_qa.answer_question import answer_question
from danswer.direct_qa.exceptions import OpenAIKeyMissing
from danswer.direct_qa.exceptions import UnknownModelError
from danswer.direct_qa.llm_utils import get_default_qa_model
from danswer.search.danswer_helper import query_intent
from danswer.search.danswer_helper import recommend_search_flow
from danswer.search.keyword_search import retrieve_keyword_documents
from danswer.search.models import QueryFlow
from danswer.search.models import SearchType
from danswer.search.semantic_search import chunks_to_search_docs
from danswer.search.semantic_search import retrieve_ranked_documents
from danswer.server.models import HelperResponse
from danswer.server.models import QAResponse
from danswer.server.models import QuestionRequest
from danswer.server.models import SearchResponse
from danswer.utils.logger import setup_logger
from danswer.utils.timing import log_generator_function_time

logger = setup_logger()

router = APIRouter()


def get_json_line(json_dict: dict) -> str:
    return json.dumps(json_dict) + "\n"


@router.get("/search-intent")
def get_search_type(
    question: QuestionRequest = Depends(), _: User = Depends(current_user)
) -> HelperResponse:
    query = question.query
    use_keyword = question.use_keyword if question.use_keyword is not None else False
    return recommend_search_flow(query, use_keyword)


@router.post("/semantic-search")
def semantic_search(
    question: QuestionRequest, user: User = Depends(current_user)
) -> SearchResponse:
    query = question.query
    collection = question.collection
    filters = question.filters
    logger.info(f"Received semantic search query: {query}")

    user_id = None if user is None else user.id
    ranked_chunks, unranked_chunks = retrieve_ranked_documents(
        query, user_id, filters, get_default_document_index(collection=collection)
    )
    if not ranked_chunks:
        return SearchResponse(top_ranked_docs=None, lower_ranked_docs=None)

    top_docs = chunks_to_search_docs(ranked_chunks)
    other_top_docs = chunks_to_search_docs(unranked_chunks)

    return SearchResponse(top_ranked_docs=top_docs, lower_ranked_docs=other_top_docs)


@router.post("/keyword-search")
def keyword_search(
    question: QuestionRequest, user: User = Depends(current_user)
) -> SearchResponse:
    query = question.query
    collection = question.collection
    filters = question.filters
    logger.info(f"Received keyword search query: {query}")

    user_id = None if user is None else user.id
    ranked_chunks = retrieve_keyword_documents(
        query, user_id, filters, get_default_document_index(collection=collection)
    )
    if not ranked_chunks:
        return SearchResponse(top_ranked_docs=None, lower_ranked_docs=None)

    top_docs = chunks_to_search_docs(ranked_chunks)
    return SearchResponse(top_ranked_docs=top_docs, lower_ranked_docs=None)


@router.post("/direct-qa")
def direct_qa(
    question: QuestionRequest, user: User = Depends(current_user)
) -> QAResponse:
    return answer_question(question=question, user=user)


@router.post("/stream-direct-qa")
def stream_direct_qa(
    question: QuestionRequest, user: User = Depends(current_user)
) -> StreamingResponse:
    send_packet_debug_msg = "Sending Packet: {}"
    top_documents_key = "top_documents"
    unranked_top_docs_key = "unranked_top_documents"
    predicted_flow_key = "predicted_flow"
    predicted_search_key = "predicted_search"

    logger.debug(f"Received QA query: {question.query}")
    logger.debug(f"Query filters: {question.filters}")
    if question.use_keyword:
        logger.debug(f"User selected Keyword Search")

    @log_generator_function_time()
    def stream_qa_portions(
        disable_generative_answer: bool = DISABLE_GENERATIVE_AI,
    ) -> Generator[str, None, None]:
        query = question.query
        collection = question.collection
        filters = question.filters
        use_keyword = question.use_keyword
        offset_count = question.offset if question.offset is not None else 0

        predicted_search, predicted_flow = query_intent(query)
        if use_keyword is None:
            use_keyword = predicted_search == SearchType.KEYWORD

        user_id = None if user is None else user.id
        if use_keyword:
            ranked_chunks: list[InferenceChunk] | None = retrieve_keyword_documents(
                query,
                user_id,
                filters,
                get_default_document_index(collection=collection),
            )
            unranked_chunks: list[InferenceChunk] | None = []
        else:
            ranked_chunks, unranked_chunks = retrieve_ranked_documents(
                query,
                user_id,
                filters,
                get_default_document_index(collection=collection),
            )
        if not ranked_chunks:
            logger.debug("No Documents Found")
            empty_docs_result = {
                top_documents_key: None,
                unranked_top_docs_key: None,
                predicted_flow_key: predicted_flow,
                predicted_search_key: predicted_search,
            }
            logger.debug(send_packet_debug_msg.format(empty_docs_result))
            yield get_json_line(empty_docs_result)
            return

        top_docs = chunks_to_search_docs(ranked_chunks)
        unranked_top_docs = chunks_to_search_docs(unranked_chunks)
        initial_response_dict = {
            top_documents_key: [top_doc.json() for top_doc in top_docs],
            unranked_top_docs_key: [doc.json() for doc in unranked_top_docs],
            # if generative AI is disabled, set flow as search so frontend
            # doesn't ask the user if they want to run QA over more documents
            predicted_flow_key: QueryFlow.SEARCH
            if disable_generative_answer
            else predicted_flow,
            predicted_search_key: predicted_search,
        }
        logger.debug(send_packet_debug_msg.format(initial_response_dict))
        yield get_json_line(initial_response_dict)

        if disable_generative_answer:
            logger.debug("Skipping QA because generative AI is disabled")
            return

        try:
            qa_model = get_default_qa_model()
        except (UnknownModelError, OpenAIKeyMissing) as e:
            logger.exception("Unable to get QA model")
            yield get_json_line({"error": str(e)})
            return

        chunk_offset = offset_count * NUM_GENERATIVE_AI_INPUT_DOCS
        if chunk_offset >= len(ranked_chunks):
            raise ValueError(
                "Chunks offset too large, should not retry this many times"
            )
        try:
            for response_packet in qa_model.answer_question_stream(
                query,
                ranked_chunks[
                    chunk_offset : chunk_offset + NUM_GENERATIVE_AI_INPUT_DOCS
                ],
            ):
                if response_packet is None:
                    continue
                logger.debug(f"Sending packet: {response_packet}")
                yield get_json_line(asdict(response_packet))
        except Exception as e:
            # exception is logged in the answer_question method, no need to re-log
            yield get_json_line({"error": str(e)})
            logger.exception("Failed to run QA")

        return

    return StreamingResponse(stream_qa_portions(), media_type="application/json")
