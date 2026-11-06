import pytest
from httpx import ASGITransport, AsyncClient

from local_rag_chatbot.api.app import create_app


@pytest.mark.asyncio
async def test_api_chat_health_and_validation_use_injected_engine(fake_engine):
    app = create_app(engine=fake_engine, mount_frontend=False)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/v1/health")
        chat = await client.post("/api/v1/chat", json={"query": "What is billing?", "conversation_id": "c-1"})
        invalid = await client.post("/api/v1/chat", json={"query": "", "conversation_id": "bad/id"})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert chat.status_code == 200
    assert chat.json()["result"] == "Answer: What is billing?"
    assert chat.json()["citations"][0]["source_name"] == "guide.md"
    assert invalid.status_code == 400
