from operator import add
from typing import Annotated, Any, Dict, List # Assurer tous les imports nécessaires

from pydantic import BaseModel

from onyx.agents.agent_search.dr.states import LoggerUpdate
# Utilise maintenant la définition corrigée de SubAgentInput (avec available_tools annoté)
from onyx.agents.agent_search.dr.sub_agents.states import SubAgentInput
from onyx.agents.agent_search.dr.models import OrchestratorTool # Importer pour FetchUpdate
from onyx.agents.agent_search.dr.sub_agents.web_search.models import (
    WebSearchResult,
)
from onyx.context.search.models import InferenceSection

# --- AJOUT ---
# Importer la fonction réducteur depuis le fichier principal dr/states.py
# (Assurez-vous qu'elle est définie là-bas comme _take_last_value_reducer)
try:
    from onyx.agents.agent_search.dr.states import _take_last_value_reducer
except ImportError:
    # Fallback si elle n'est pas définie/trouvée (ajustez si nécessaire)
    def _take_last_value_reducer(existing_value: Any, new_value: Any) -> Any:
        return new_value
# --- FIN AJOUT ---


class InternetSearchInput(SubAgentInput):
    results_to_open: Annotated[list[tuple[str, WebSearchResult]], add] = []
    parallelization_nr: int = 0
    branch_question: Annotated[str, lambda x, y: y] = "" # Ce champ est OK ici
    branch_questions_to_urls: Annotated[dict[str, list[str]], lambda x, y: y] = {}
    raw_documents: Annotated[list[InferenceSection], add] = []
    url_to_raw_document_map: Annotated[dict[str, InferenceSection], lambda x, y: y] = {} # Assurez-vous qu'il est bien là


class InternetSearchUpdate(LoggerUpdate):
    results_to_open: Annotated[list[tuple[str, WebSearchResult]], add] = []


class FetchInput(SubAgentInput):
    urls_to_open: Annotated[list[str], add] = []
    branch_questions_to_urls: Dict[str, List[str]]
    raw_documents: Annotated[list[InferenceSection], add] = []
    branch_question: str


class FetchUpdate(LoggerUpdate):
    raw_documents: Annotated[list[InferenceSection], add] = []
    branch_questions_to_urls: Annotated[Dict[str, List[str]], lambda x, y: y] = {}
    branch_question: Annotated[str, lambda x, y: y] = ""
    tools_used: Annotated[List[str], lambda x, y: y] = []

    # Utilise l'annotation cohérente importée ou définie ci-dessus
    iteration_nr: Annotated[int, _take_last_value_reducer]

    url_to_raw_document_map: Annotated[dict[str, InferenceSection], lambda x, y: y] = {}

    # --- VÉRIFICATION ---
    # S'assurer que cette ligne utilise EXACTEMENT la même annotation que SubAgentInput
    available_tools: Annotated[dict[str, OrchestratorTool] | None, lambda x, y: y] = None
    # --- FIN VÉRIFICATION ---


class SummarizeInput(SubAgentInput):
    raw_documents: Annotated[list[InferenceSection], add] = []
    branch_questions_to_urls: Dict[str, List[str]]
    branch_question: str
    url_to_raw_document_map: Dict[str, InferenceSection]