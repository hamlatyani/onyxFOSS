from onyx.agents.agent_search.deep_search_a.initial.generate_individual_sub_answer.states import (
    RetrievalIngestionUpdate,
)
from onyx.agents.agent_search.deep_search_a.shared.expanded_retrieval.states import (
    ExpandedRetrievalOutput,
)
from onyx.agents.agent_search.shared_graph_utils.models import AgentChunkStats


def ingest_retrieval(state: ExpandedRetrievalOutput) -> RetrievalIngestionUpdate:
    sub_question_retrieval_stats = (
        state.expanded_retrieval_result.sub_question_retrieval_stats
    )
    if sub_question_retrieval_stats is None:
        sub_question_retrieval_stats = [AgentChunkStats()]

    return RetrievalIngestionUpdate(
        expanded_retrieval_results=state.expanded_retrieval_result.expanded_queries_results,
        documents=state.expanded_retrieval_result.reranked_documents,
        context_documents=state.expanded_retrieval_result.context_documents,
        sub_question_retrieval_stats=sub_question_retrieval_stats,
    )
