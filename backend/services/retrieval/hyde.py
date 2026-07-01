import os
import re
import time
import groq
from config import get_settings
from services.generation.prompts import HYDE_PROMPT
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


def generate_hypothetical_excerpt(question: str) -> str:
    """
    Generates a hypothetical policy excerpt that would answer the question
    using the Groq model (llama-3.3-70b-versatile).
    Includes smart rate limit parsing and exponential backoff retries.
    If generation fails after retries, logs the exception and falls back to the original question.
    """
    api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("groq_api_key_missing_hyde_fallback", question=question)
        return question

    client = groq.Groq(api_key=api_key)
    backoff = 2.0
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            prompt = HYDE_PROMPT.format(question=question)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )

            if response.choices and response.choices[0].message.content:
                excerpt = response.choices[0].message.content.strip()
                logger.info("hyde_excerpt_generated", question=question, excerpt_length=len(excerpt))
                return excerpt
            else:
                logger.warning("hyde_empty_response", question=question)
                return question

        except groq.RateLimitError as e:
            if attempt == max_attempts - 1:
                logger.exception("hyde_generation_failed_exhausted", question=question, error=str(e))
                return question

            err_msg = str(e)
            wait_seconds = backoff
            match = re.search(r"(?:retry|try again) in ([0-9.]+)(m?s)", err_msg, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit == "ms":
                    wait_seconds = (value / 1000.0) + 1.5
                else:
                    wait_seconds = value + 1.5
            else:
                time_str_match = re.search(r"in\s+([0-9hm\.]+s)", err_msg, re.IGNORECASE)
                if time_str_match:
                    time_str = time_str_match.group(1).rstrip(".")
                    if "m" in time_str:
                        parts = time_str.split("m")
                        minutes = float(parts[0])
                        seconds = float(parts[1].replace("s", ""))
                        wait_seconds = minutes * 60.0 + seconds + 1.5
                    else:
                        seconds = float(time_str.replace("s", ""))
                        wait_seconds = seconds + 1.5
                else:
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
            if getattr(e, 'status_code', None) == 429:
                if attempt == max_attempts - 1:
                    logger.exception("hyde_generation_failed_exhausted", question=question, error=str(e))
                    return question
                time.sleep(backoff)
                backoff *= 2.0
                continue

            logger.exception("hyde_generation_failed", question=question, error=str(e))
            return question

    return question

