import abc
from collections.abc import Generator
from dataclasses import dataclass

from danswer.chunking.models import InferenceChunk


@dataclass
class DanswerAnswer:
    answer: str | None


@dataclass
class DanswerAnswerPiece:
    """A small piece of a complete answer. Used for streaming back answers."""

    answer_piece: str | None  # if None, specifies the end of an Answer


@dataclass
class DanswerQuote:
    # This is during inference so everything is a string by this point
    quote: str
    document_id: str
    link: str | None
    source_type: str
    semantic_identifier: str
    blurb: str


@dataclass
class DanswerQuotes:
    """A little clunky, but making this into a separate class so that the result from
    `answer_question_stream` is always a subclass of `dataclass` and can thus use `asdict()`
    """

    quotes: list[DanswerQuote]


AnswerQuestionReturn = tuple[DanswerAnswer, DanswerQuotes]
AnswerQuestionStreamReturn = Generator[
    DanswerAnswerPiece | DanswerQuotes | None, None, None
]


class QAModel:
    @property
    def requires_api_key(self) -> bool:
        """Is this model protected by security features
        Does it need an api key to access the model for inference"""
        return True

    def warm_up_model(self) -> None:
        """This is called during server start up to load the models into memory
        pass if model is accessed via API"""
        pass

    @abc.abstractmethod
    def answer_question(
        self,
        query: str,
        context_docs: list[InferenceChunk],
    ) -> AnswerQuestionReturn:
        raise NotImplementedError

    @abc.abstractmethod
    def answer_question_stream(
        self,
        query: str,
        context_docs: list[InferenceChunk],
    ) -> AnswerQuestionStreamReturn:
        raise NotImplementedError
