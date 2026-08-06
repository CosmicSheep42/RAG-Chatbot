const messages = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("query");
const sendButton = document.getElementById("send-button");
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const inputCounter = document.getElementById("input-counter");
const featureList = document.getElementById("feature-list");
const rerankingNote = document.getElementById("reranking-note");

let statusPollTimer = null;
let maxInputChars = 1000;
const conversationStorageKey = "chat_conversation_id";

function createConversationId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  if (window.crypto && typeof window.crypto.getRandomValues === "function") {
    const values = new Uint32Array(4);
    window.crypto.getRandomValues(values);
    return Array.from(values, (value) => value.toString(16).padStart(8, "0")).join("-");
  }

  return null;
}

function getConversationId() {
  try {
    let conversationId = window.sessionStorage.getItem(conversationStorageKey);

    if (!conversationId) {
      conversationId = createConversationId();
      if (conversationId) {
        window.sessionStorage.setItem(conversationStorageKey, conversationId);
      }
    }

    return conversationId;
  } catch (_error) {
    return null;
  }
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function setStatus(kind, text) {
  statusBadge.classList.remove("status-warm", "status-ready", "status-error");
  statusBadge.classList.add(`status-${kind}`);

  if (kind === "ready") {
    statusBadge.textContent = "Ready";
  } else if (kind === "error") {
    statusBadge.textContent = "Error";
  } else {
    statusBadge.textContent = "Starting";
  }

  statusText.textContent = text;
}

function setComposerEnabled(enabled) {
  input.disabled = !enabled;
  sendButton.disabled = !enabled;
}

function enabledText(value) {
  return value ? "active" : "off";
}

function describeReranking(reranking) {
  if (!reranking || reranking.status === "initializing") {
    return {
      label: "checking",
      note: "",
    };
  }

  if (reranking.active) {
    return {
      label: "active",
      note: "Reranking uses an additional model to reorganize retrieved document passages and can improve difficult or ambiguous searches.",
    };
  }

  if (reranking.status === "disabled") {
    return {
      label: "disabled by configuration",
      note: "Reranking was disabled by configuration. The chatbot is using standard hybrid retrieval.",
    };
  }

  if (reranking.status === "not_applicable") {
    return {
      label: "waiting for indexed documents",
      note: "Reranking will be checked after documents are indexed.",
    };
  }

  if (reranking.status === "dependency_missing") {
    return {
      label: "optional feature unavailable",
      note: [
        "Reranking is not installed.",
        "The chatbot is using standard hybrid retrieval.",
        "Reranking uses an additional model to reorganize retrieved document passages and can improve difficult or ambiguous searches.",
        'Install with: pip install -e ".[rerank]"',
      ].join(" "),
    };
  }

  if (reranking.status === "load_failed") {
    return {
      label: "unavailable",
      note: "The reranker model could not be loaded. The chatbot is using standard hybrid retrieval.",
    };
  }

  return {
    label: "off",
    note: reranking.reason || "",
  };
}

function updateFeatureStatus(data) {
  const features = data.features || {};
  const reranking = describeReranking(features.reranking);
  featureList.innerHTML = "";

  [
    `Memory: ${enabledText(Boolean(features.memory))}`,
    `Query rewriting: ${enabledText(Boolean(features.query_rewriting))}`,
    `Tool calling: ${enabledText(Boolean(features.tool_calling))}`,
    `Reranking: ${reranking.label}`,
  ].forEach((text) => {
    const item = document.createElement("div");
    item.textContent = text;
    featureList.appendChild(item);
  });

  rerankingNote.textContent = reranking.note;
  rerankingNote.hidden = !reranking.note;
}

function describeCitation(citation) {
  const parts = [citation.citation_id || "source"];
  if (citation.source_name || citation.source) {
    parts.push(citation.source_name || citation.source);
  }
  if (citation.page_number) parts.push(`page ${citation.page_number}`);
  if (citation.slide_number) parts.push(`slide ${citation.slide_number}`);
  if (citation.sheet_name) parts.push(`sheet ${citation.sheet_name}`);
  if (citation.section) parts.push(citation.section);
  return parts.join(" · ");
}

function appendCopyButton(article, text) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "copy-button";
  button.textContent = "Copy";
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(String(text ?? ""));
      button.textContent = "Copied";
      setTimeout(() => {
        button.textContent = "Copy";
      }, 1200);
    } catch (_error) {
      button.textContent = "Copy failed";
    }
  });
  article.appendChild(button);
}

function setInputLimit(limit) {
  const parsed = Number(limit);
  if (Number.isFinite(parsed) && parsed > 0) {
    maxInputChars = parsed;
  }

  input.maxLength = maxInputChars;
  updateInputCounter();
}

function updateInputCounter() {
  const currentLength = input.value.length;
  inputCounter.textContent = `${currentLength} / ${maxInputChars}`;

  if (currentLength > maxInputChars * 0.9) {
    inputCounter.style.color = "#fde68a";
  } else {
    inputCounter.style.color = "";
  }
}

function renderMessage(role, text, extra = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const content = document.createElement("div");
  content.textContent = String(text ?? "");
  article.appendChild(content);

  if (role === "assistant") {
    appendCopyButton(article, text);
  }

  if (extra.meta) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = extra.meta;
    article.appendChild(meta);
  }

  if (Array.isArray(extra.toolCalls) && extra.toolCalls.length > 0) {
    const tools = document.createElement("details");
    tools.className = "sources";

    const summary = document.createElement("summary");
    summary.textContent = `Tool calls (${extra.toolCalls.length})`;
    tools.appendChild(summary);

    const toolList = document.createElement("div");
    toolList.className = "source-list";

    extra.toolCalls.forEach((toolCall) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const query = toolCall?.arguments?.query;
      item.textContent = query
        ? `${toolCall.name}: ${query}`
        : `${toolCall.name}`;
      toolList.appendChild(item);
    });

    tools.appendChild(toolList);
    article.appendChild(tools);
  }

  if (Array.isArray(extra.citations) && extra.citations.length > 0) {
    const sources = document.createElement("details");
    sources.className = "sources";

    const summary = document.createElement("summary");
    summary.textContent = `Sources (${extra.citations.length})`;
    sources.appendChild(summary);

    const sourceList = document.createElement("div");
    sourceList.className = "source-list";

    extra.citations.forEach((source) => {
      const item = document.createElement("div");
      item.className = "source-item";
      const label = document.createElement("div");
      label.textContent = describeCitation(source);
      const snippet = document.createElement("div");
      snippet.textContent = source.snippet || "";
      item.appendChild(label);
      item.appendChild(snippet);
      sourceList.appendChild(item);
    });

    sources.appendChild(sourceList);
    article.appendChild(sources);
  }

  messages.appendChild(article);
  scrollToBottom();
}

function renderSystem(text) {
  renderMessage("system", text);
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/v1/status", { cache: "no-store" });
    const data = await response.json();

    if (data.ready) {
      const ragStatus = data.rag && data.rag.status;
      const statusText = ragStatus === "empty"
        ? "No documents indexed yet."
        : "System ready";
      setStatus("ready", statusText);
      updateFeatureStatus(data);
      if (data.controls && data.controls.max_input_chars) {
        setInputLimit(data.controls.max_input_chars);
      }
      setComposerEnabled(true);
      if (statusPollTimer) {
        clearInterval(statusPollTimer);
        statusPollTimer = null;
      }
      return;
    }

    if (data.initializing) {
      setStatus("warm", "Loading documents, embeddings, and router.");
      updateFeatureStatus(data);
      if (data.controls && data.controls.max_input_chars) {
        setInputLimit(data.controls.max_input_chars);
      }
      setComposerEnabled(false);
      return;
    }

    setStatus("error", data.error || "Initialization is not complete yet.");
    updateFeatureStatus(data);
    if (data.controls && data.controls.max_input_chars) {
      setInputLimit(data.controls.max_input_chars);
    }
    setComposerEnabled(true);
  } catch (error) {
    setStatus("error", "Could not read backend status.");
    setComposerEnabled(true);
  }
}

async function sendMessage(query) {
  const conversationId = getConversationId();
  const body = { query };
  if (conversationId) {
    body.conversation_id = conversationId;
  }

  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  const data = await response.json();

  if (!response.ok) {
    const retryAfter = response.headers.get("Retry-After");
    const suffix = retryAfter ? ` Retry after ${retryAfter} seconds.` : "";
    throw new Error((data.error || "Failed to process the message.") + suffix);
  }

  return data;
}

function formatMeta(data) {
  const parts = [];

  if (data.route) {
    parts.push(`route: ${data.route}`);
  }

  if (data.matched_category && data.matched_category !== data.route) {
    parts.push(`category: ${data.matched_category}`);
  }

  if (data.matched_intent) {
    parts.push(`intent: ${data.matched_intent}`);
  }

  if (typeof data.score === "number") {
    parts.push(`score: ${data.score.toFixed(3)}`);
  }

  if (data.reason) {
    parts.push(`reason: ${data.reason}`);
  }

  if (data.tool_calling_used) {
    const count = Array.isArray(data.tool_calls) ? data.tool_calls.length : 0;
    parts.push(`tools: ${count}`);
  }

  return parts.join(" · ");
}

async function handleSubmit(event) {
  event.preventDefault();

  const query = input.value.trim();
  if (!query) {
    return;
  }

  input.value = "";
  renderMessage("user", query);
  setComposerEnabled(false);
  setStatus("warm", "Querying the router and RAG.");

  try {
    const data = await sendMessage(query);
    renderMessage("assistant", data.result || "No response returned.", {
      meta: formatMeta(data),
      toolCalls: data.tool_calls || [],
      citations: data.citations || [],
    });
    setStatus("ready", "Response delivered by the local backend.");
  } catch (error) {
    renderMessage("system", error.message || "Failed to send the message.");
    setStatus("error", error.message || "Failed to send the message.");
  } finally {
    setComposerEnabled(true);
    input.focus();
  }
}

function bootstrap() {
  renderSystem("The chat is ready. Messages stay inside the panel and the page does not grow.");
  setInputLimit(maxInputChars);
  input.addEventListener("input", updateInputCounter);
  fetchStatus();
  statusPollTimer = window.setInterval(fetchStatus, 1500);
  form.addEventListener("submit", handleSubmit);
  input.focus();
}

bootstrap();
