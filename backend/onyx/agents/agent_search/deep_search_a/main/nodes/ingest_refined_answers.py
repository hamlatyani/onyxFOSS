from datetime import datetime

from onyx.agents.agent_search.deep_search_a.initial.generate_individual_sub_answer.states import (
    AnswerQuestionOutput,
)
from onyx.agents.agent_search.deep_search_a.main.operations import logger
from onyx.agents.agent_search.deep_search_a.main.states import (
    DecompAnswersUpdate,
)
from onyx.agents.agent_search.shared_graph_utils.operators import (
    dedup_inference_sections,
)


def ingest_refined_answers(
    state: AnswerQuestionOutput,
) -> DecompAnswersUpdate:
    now_start = datetime.now()

    logger.info(f"--------{now_start}--------INGEST FOLLOW UP ANSWERS---")

    documents = []
    answer_results = state.answer_results if hasattr(state, "answer_results") else []
    for answer_result in answer_results:
        documents.extend(answer_result.documents)

    now_end = datetime.now()

    logger.debug(
        f"--------{now_end}--{now_end - now_start}--------INGEST FOLLOW UP ANSWERS END---"
    )

    return DecompAnswersUpdate(
        # Deduping is done by the documents operator for the main graph
        # so we might not need to dedup here
        documents=dedup_inference_sections(documents, []),
        sub_question_results=answer_results,
        log_messages=[
            f"{now_start} -- Main - Ingest refined answers,  Time taken: {now_end - now_start}"
        ],
    )
