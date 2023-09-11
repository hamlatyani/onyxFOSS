from danswer.direct_qa.qa_block import dict_based_prompt_to_langchain_prompt
from danswer.direct_qa.qa_prompts import ANSWER_PAT
from danswer.direct_qa.qa_prompts import CODE_BLOCK_PAT
from danswer.direct_qa.qa_prompts import GENERAL_SEP_PAT
from danswer.direct_qa.qa_prompts import INVALID_PAT
from danswer.direct_qa.qa_prompts import QUESTION_PAT
from danswer.direct_qa.qa_prompts import THOUGHT_PAT
from danswer.llm.build import get_default_llm
from danswer.utils.logger import setup_logger
from danswer.utils.timing import log_function_time

logger = setup_logger()


def get_answer_validation_messages(query: str, answer: str) -> list[dict[str, str]]:
    cot_block = (
        f"{THOUGHT_PAT} Use this as a scratchpad to write out in a step by step manner your reasoning "
        f"about EACH criterion to ensure that your conclusion is correct.\n"
        f"{INVALID_PAT} True or False"
    )

    q_a_block = f"{QUESTION_PAT} {query}\n\n" f"{ANSWER_PAT} {answer}"

    messages = [
        {
            "role": "user",
            "content": (
                f"{CODE_BLOCK_PAT.format(q_a_block).lstrip()}{GENERAL_SEP_PAT}\n"
                "Determine if the answer is valid for the query.\n"
                f"The answer is invalid if ANY of the following is true:\n"
                "- Does not directly answer the user query.\n"
                "- Answers a related but different question.\n"
                '- Contains anything meaning "I don\'t know" or "information not found".\n\n'
                f"You must use the following format:"
                f"{CODE_BLOCK_PAT.format(cot_block)}"
                f'Hint: Invalid must be exactly "True" or "False" (without the quotes)'
            ),
        },
    ]

    return messages


def extract_validity(model_output: str) -> bool:
    if INVALID_PAT in model_output:
        result = model_output.split(INVALID_PAT)[-1].strip()
        if "true" in result.lower():
            return False
    return True  # If something is wrong, let's not toss away the answer


@log_function_time()
def get_answer_validity(
    query: str,
    answer: str,
) -> bool:
    messages = get_answer_validation_messages(query, answer)
    filled_llm_prompt = dict_based_prompt_to_langchain_prompt(messages)
    model_output = get_default_llm().invoke(filled_llm_prompt)
    logger.debug(model_output)

    validity = extract_validity(model_output)
    logger.info(
        f'LLM Answer of "{answer}" was determined to be {"valid" if validity else "invalid"}.'
    )

    return validity
