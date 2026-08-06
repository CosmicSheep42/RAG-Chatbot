from __future__ import annotations

import logging
import sys

from local_rag_chatbot.observability import configure_logging
from local_rag_chatbot.settings import get_settings


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    try:
        from local_rag_chatbot.application import create_indexing_services

        services = create_indexing_services(settings)
        logger.info("Preparing RAG indexes")
        ok = services.index_manager.prepare()
        status = services.index_manager.status()
        if not ok:
            logger.error("Index preparation did not complete", extra={"rag_status": status})
            return 1
        logger.info("Index preparation completed", extra={"rag_status": status})
        return 0
    except Exception:
        logger.exception("Index preparation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
