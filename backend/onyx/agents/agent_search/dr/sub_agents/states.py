from operator import add
# --- AJOUT ---
# Importer Annotated, Any, Dict, List pour les corrections
from typing import Annotated, Any, Dict, List
# --- FIN AJOUT ---

from pydantic import BaseModel # Import nécessaire

from onyx.agents.agent_search.dr.models import IterationAnswer
from onyx.agents.agent_search.dr.models import OrchestratorTool # Assurez-vous que c'est importé
from onyx.agents.agent_search.dr.states import LoggerUpdate # LoggerUpdate est bien importé
from onyx.db.connector import DocumentSource
# --- AJOUT ---
# Importer InferenceSection pour les types manquants
from onyx.context.search.models import InferenceSection
# --- FIN AJOUT ---


class SubAgentUpdate(LoggerUpdate):
    iteration_responses: Annotated[list[IterationAnswer], add] = []
    current_step_nr: Annotated[int, lambda x, y: y] = 1


class BranchUpdate(LoggerUpdate):
    branch_iteration_responses: Annotated[list[IterationAnswer], add] = []


class SubAgentInput(LoggerUpdate):
    iteration_nr: Annotated[int, lambda x, y: y] = 0
    current_step_nr: Annotated[int, lambda x, y: y] = 1
    query_list: Annotated[list[str], lambda x, y: y] = []
    context: str | None = None
    active_source_types: list[DocumentSource] | None = None
    tools_used: Annotated[list[str], add] = []

    # --- CORRECTION ---
    # Ajouter l'annotation pour available_tools
    available_tools: Annotated[dict[str, OrchestratorTool] | None, lambda x, y: y] = None
    # --- FIN CORRECTION ---

    assistant_system_prompt: str | None = None
    assistant_task_prompt: str | None = None


class SubAgentMainState(
    SubAgentInput,
    SubAgentUpdate,
    BranchUpdate,
):
    # Les champs ajoutés précédemment sont corrects car ils héritent implicitement
    # les définitions (ou leurs annotations) de SubAgentInput ou des classes Update.
    branch_questions_to_urls: Annotated[Dict[str, List[str]], lambda x, y: y] = {}
    raw_documents: Annotated[list[InferenceSection], add] = []
    url_to_raw_document_map: Annotated[Dict[str, InferenceSection], lambda x, y: y] = {}
    results_to_open: Annotated[list[tuple[str, Any]], add] = [] # Utiliser Any pour WebSearchResult si non importé ici
    pass # Garder 'pass' si rien d'autre n'est ajouté ici


# Utilise maintenant la définition corrigée de SubAgentInput
class BranchInput(SubAgentInput):
    parallelization_nr: int = 0
    branch_question: str


# Hérite de LoggerUpdate pour inclure log_messages
class CustomToolBranchInput(LoggerUpdate):
    tool_info: OrchestratorTool