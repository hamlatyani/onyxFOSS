from pydantic import BaseModel


class FollowUpSubQuestion(BaseModel):
    sub_question: str
    sub_question_id: str
    verified: bool
    answered: bool
    answer: str


class AgentTimings(BaseModel):
    base_duration__s: float | None
    refined_duration__s: float | None
    full_duration__s: float | None


class AgentBaseMetrics(BaseModel):
    num_verified_documents_total: int | None
    num_verified_documents_core: int | None
    verified_avg_score_core: float | None
    num_verified_documents_base: int | float | None
    verified_avg_score_base: float | None
    base_doc_boost_factor: float | None
    support_boost_factor: float | None
    duration__s: float | None


class AgentRefinedMetrics(BaseModel):
    refined_doc_boost_factor: float | None
    refined_question_boost_factor: float | None
    duration__s: float | None


class AgentAdditionalMetrics(BaseModel):
    pass
