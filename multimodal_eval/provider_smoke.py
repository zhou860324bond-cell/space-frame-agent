"""One-image provider connectivity/schema smoke tests; never part of offline score."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from multimodal_contract import file_digest
from multimodal_workflow import MultimodalControllerState
from sketch_parser import SketchParser

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def run(*, online: bool = False) -> dict:
    image = ROOT / "images" / "seed_01_portal.png"
    results = []
    for provider, env_var in PROVIDERS.items():
        if not os.environ.get(env_var):
            results.append({"provider": provider,
                            "status": "SKIPPED_NO_CREDENTIALS"})
            continue
        if not online:
            results.append({"provider": provider,
                            "status": "SKIPPED_ONLINE_NOT_REQUESTED"})
            continue
        state = MultimodalControllerState()
        state.load_image(file_digest(image))
        job = state.start_recognition(f"smoke-{provider}")
        parsed = SketchParser.from_env(provider).parse_v2_with_retry(
            image, state, job, max_repairs=0)
        results.append({
            "provider": provider,
            "model": SketchParser._default_model(provider),
            "status": "PASS" if parsed.success else "FAIL",
            "attempts": parsed.attempts,
            "errors": parsed.errors,
        })
    return {
        "format": "space-frame-provider-smoke/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "image_id": "seed_01_portal", "excluded_from_offline_gate": True,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "provider_smoke.json"))
    args = parser.parse_args()
    report = run(online=args.online)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if any(item["status"] == "FAIL" for item in report["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
