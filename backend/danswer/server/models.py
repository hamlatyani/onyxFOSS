from danswer.datastores.interfaces import DatastoreFilter
from pydantic import BaseModel


class UserRoleResponse(BaseModel):
    role: str


class SearchDoc(BaseModel):
    semantic_name: str
    link: str
    blurb: str
    source_type: str


class QAQuestion(BaseModel):
    query: str
    collection: str
    filters: list[DatastoreFilter] | None


class QAResponse(BaseModel):
    answer: str | None
    quotes: dict[str, dict[str, str | int | None]] | None
    ranked_documents: list[SearchDoc] | None


class KeywordResponse(BaseModel):
    results: list[str] | None


class UserByEmail(BaseModel):
    user_email: str
