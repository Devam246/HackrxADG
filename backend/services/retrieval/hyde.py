import re
import time
import google.generativeai as genai
import google.api_core.exceptions
from config import get_settings
from services.generation.prompts import HYDE_PROMPT
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


def generate_hypothetical_excerpt(question: str) -> str:
    """
    Generates a hypothetical policy excerpt that would answer the question
    using the Gemini model configured in Settings.
    Includes smart rate limit parsing and exponential backoff retries.
    If generation fails after retries, logs the exception and falls back to the original question.
    """
    backoff = 2.0
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel(settings.llm_model)

            prompt = HYDE_PROMPT.format(question=question)

            response = model.generate_content(
                contents=[prompt],
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 200,
                },
            )
            if response.text:
                excerpt = response.text.strip()
                logger.info("hyde_excerpt_generated", question=question, excerpt_length=len(excerpt))
                return excerpt
            else:
                logger.warning("hyde_empty_response", question=question)
                return question

        except google.api_core.exceptions.ResourceExhausted as e:
            # Catch 429 quota/rate limit error specifically
            if attempt == max_attempts - 1:
                logger.exception("hyde_generation_failed_exhausted", question=question, error=str(e))
                return question

            # Parse required wait time from the rate limit exception
            err_msg = str(e)
            match = re.search(r"Please retry in ([0-9.]+)s", err_msg)
            if match:
                wait_seconds = float(match.group(1)) + 1.5  # Add a buffer
            else:
                wait_seconds = backoff
                backoff *= 2.0

            logger.warning(
                "hyde_rate_limit_hit_retrying",
                question=question,
                attempt=attempt + 1,
                wait_seconds=round(wait_seconds, 2),
                error=err_msg,
            )
            time.sleep(wait_seconds)

        except Exception as e:
            # Fall back immediately on other exceptions
            logger.exception("hyde_generation_failed", question=question, error=str(e))
            return question

    return question
