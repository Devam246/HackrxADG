import spacy
import structlog


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


logger = structlog.get_logger(__name__)


def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spacy_model_missing", fallback="blank_en")
        return spacy.blank("en")
