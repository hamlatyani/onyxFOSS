from datetime import datetime
from typing import cast

from langchain_core.runnables import RunnableConfig

from onyx.agents.agent_search.deep_search_a.main__graph.operations import logger
from onyx.agents.agent_search.deep_search_a.main__graph.states import (
    ExploratorySearchUpdate,
)
from onyx.agents.agent_search.deep_search_a.main__graph.states import MainState
from onyx.agents.agent_search.models import AgentSearchConfig
from onyx.agents.agent_search.shared_graph_utils.agent_prompt_ops import (
    build_history_prompt,
)
from onyx.agents.agent_search.shared_graph_utils.utils import retrieve_search_docs
from onyx.configs.agent_configs import AGENT_EXPLORATORY_SEARCH_RESULTS
from onyx.context.search.models import InferenceSection


def agent_search_start(
    state: MainState, config: RunnableConfig
) -> ExploratorySearchUpdate:
    now_start = datetime.now()

    logger.info(f"--------{now_start}--------EXPLORATORY SEARCH START---")

    agent_a_config = cast(AgentSearchConfig, config["metadata"]["config"])
    question = agent_a_config.search_request.query
    chat_session_id = agent_a_config.chat_session_id
    primary_message_id = agent_a_config.message_id
    agent_a_config.fast_llm

    history = build_history_prompt(agent_a_config, question)

    if chat_session_id is None or primary_message_id is None:
        raise ValueError(
            "chat_session_id and message_id must be provided for agent search"
        )

    # Initial search to inform decomposition. Just get top 3 fits

    search_tool = agent_a_config.search_tool
    if search_tool is None:
        raise ValueError("search_tool must be provided for agentic search")
    retrieved_docs: list[InferenceSection] = retrieve_search_docs(search_tool, question)

    exploratory_search_results = retrieved_docs[:AGENT_EXPLORATORY_SEARCH_RESULTS]
    now_end = datetime.now()
    logger.debug(
        f"--------{now_end}--{now_end - now_start}--------EXPLORATORY SEARCH END---"
    )

    return ExploratorySearchUpdate(
        exploratory_search_results=exploratory_search_results,
        previous_history_summary=history,
        log_messages=[
            f"{now_start} -- Main - Exploratory Search,  Time taken: {now_end - now_start}"
        ],
    )
