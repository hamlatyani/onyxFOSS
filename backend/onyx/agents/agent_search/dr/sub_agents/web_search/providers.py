from onyx.agents.agent_search.dr.sub_agents.web_search.clients.exa_client import (
    ExaClient,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.clients.google_client import (
    GoogleClient,
)

from onyx.agents.agent_search.dr.sub_agents.web_search.clients.serper_client import (
    SerperClient,
)
from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    WebSearchProvider,
)
from onyx.configs.chat_configs import EXA_API_KEY, SERPER_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID


def get_default_provider() -> WebSearchProvider | None:
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            return GoogleClient()
        except ValueError:
            pass
    if EXA_API_KEY:
        return ExaClient()
    
    if SERPER_API_KEY:
        return SerperClient()
    return None