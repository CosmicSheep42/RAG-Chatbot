from __future__ import annotations

import uvicorn

from local_rag_chatbot.config import ChatbotConfig
from local_rag_chatbot.fastapi import create_app

config = ChatbotConfig.from_env()
app = create_app(config=config)


def main() -> None:
    uvicorn.run(app, host=config.server_host, port=config.server_port)


if __name__ == "__main__":
    main()
