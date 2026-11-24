"""Explicit live inference, MinerU parsing or single SMTP acceptance probe."""

import argparse
import os
from pathlib import Path

import httpx
from email_validator import validate_email

from med_evidence_agentic_rag.evaluation.load import LoadConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=["model", "parser", "smtp"])
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--target", required=True)
    parser.add_argument("--recipient")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if not args.allow_live:
        parser.error("--allow-live required")
    try:
        if args.service == "model":
            origin = LoadConfig(target=args.target, allow_live=True)
            url = origin.target.rstrip("/") + "/v1/chat/completions"
            response = httpx.post(
                url,
                headers={"Authorization": "Bearer " + os.environ["LLM_API_KEY"]},
                json={
                    "model": os.environ["LLM_MODEL_NAME"],
                    "messages": [{"role": "user", "content": "Reply with the word probe."}],
                    "max_tokens": 16,
                },
                timeout=30,
            )
            ok = bool(response.raise_for_status().json()["choices"][0]["message"]["content"])
        elif args.service == "parser":
            if args.target != "mineru-official-sdk" or not args.input or not args.input.is_file():
                parser.error("Select mineru-official-sdk and an explicit input file")
            from med_evidence_agentic_rag.rag_creation.mineru_loader import MinerUClient

            ok = bool(MinerUClient(os.environ["MINERU_TOKEN"]).parse(args.input).strip())
        else:
            if args.target != os.environ.get("SMTP_HOST") or not args.recipient:
                parser.error("Match the configured SMTP_HOST and select a recipient")
            recipient = validate_email(args.recipient, check_deliverability=False).normalized
            from med_evidence_agentic_rag.tools.send_email import send_email

            ok = send_email(recipient, "Verification", "Explicitly requested delivery probe.")
    except Exception:
        print("Probe failed or outcome uncertain")
        return 1
    print("Probe succeeded" if ok else "Probe failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
