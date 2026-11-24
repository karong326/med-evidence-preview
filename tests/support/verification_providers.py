"""Loopback-only protocol fixtures for the disposable deployment."""

import json
import os
import sys
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import ModuleType
from uuid import uuid4

QUERY = "Synthetic marker lookup"
PASSAGE = "The synthetic marker is cobalt."
MAIL = "Create a synthetic mail draft"


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, content, mime="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content.encode())))
        self.end_headers()
        self.wfile.write(content.encode())

    def do_GET(self):
        assert self.path == "/v1/models"
        self.respond(json.dumps({"data": [{"id": os.environ["LLM_MODEL_NAME"]}]}))

    def do_POST(self):
        assert self.path == "/v1/chat/completions"
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body["messages"]
        tools = [tool["function"]["name"] for tool in body.get("tools", [])]
        message = {"role": "assistant"}
        chosen = None
        if tools == ["QueryPlan"]:
            chosen = ("QueryPlan", {"mode": "single", "queries": [QUERY], "fallback": False})
        elif messages[-1]["role"] == "user":
            if MAIL in messages[-1]["content"]:
                chosen = (
                    "send_email",
                    {
                        "recipient": "capture@example.org",
                        "email_subject": "fixture",
                        "email_content": "synthetic",
                    },
                )
            else:
                chosen = ("rewrite_queries", {"question": QUERY})
        else:
            result = json.loads(messages[-1]["content"])
            if "queries" in result:
                assert result["fallback"] is False
                chosen = ("search_knowledge", {"query": result["queries"][0]})
            elif "evidences" in result:
                evidence = result["evidences"][0]
                message["content"] = f"{evidence['content']} [{evidence['citation_id']}]"
            else:
                assert result["requires_confirmation"]
                message["content"] = "Draft ready."
        if chosen:
            name, args = chosen
            assert name in tools
            message["tool_calls"] = [
                {
                    "id": "call_" + uuid4().hex,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        base = {"id": "chatcmpl-fixture", "model": os.environ["LLM_MODEL_NAME"], "created": 1}
        reason = "tool_calls" if chosen else "stop"
        if body.get("stream"):
            if chosen:
                message["tool_calls"][0]["index"] = 0
            chunks = [
                {
                    **base,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                for delta, finish in [(message, None), ({}, reason)]
            ]
            self.respond(
                "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n",
                "text/event-stream",
            )
        else:
            self.respond(
                json.dumps(
                    {
                        **base,
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
                    }
                )
            )


@contextmanager
def model_endpoint():
    with ThreadingHTTPServer(("127.0.0.1", 0), Provider) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            thread.join()


class Capture:
    def __init__(self):
        self.accepted = []

    async def session(self, reader, writer):
        try:
            writer.write(b"220 fixture ESMTP\r\n")
            while line := await reader.readline():
                command = line.split()[0].upper()
                if command == b"QUIT":
                    writer.write(b"221 bye\r\n")
                    break
                if command == b"DATA":
                    writer.write(b"354 end with dot\r\n")
                    await writer.drain()
                    payload = await reader.readuntil(b"\r\n.\r\n")
                    self.accepted.append(payload)
                writer.write(b"250 OK\r\n")
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def install_inference():
    from langchain_core.embeddings import Embeddings

    class FixedEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0]

    class FixedReranker:
        def predict(self, pairs, **kwargs):
            return [1.0] * len(pairs)

    for name, attr, value in [
        ("embedding", "embeddings", FixedEmbeddings()),
        ("reranker", "reranker", FixedReranker()),
    ]:
        key = "med_evidence_agentic_rag.models_and_database." + name
        module = ModuleType(key)
        setattr(module, attr, value)
        sys.modules[key] = module
