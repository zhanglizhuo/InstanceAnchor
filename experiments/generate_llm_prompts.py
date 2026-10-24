"""
generate_llm_prompts.py — Fill CAPE_LLM_GENERATED using an LLM API.
No GPU needed. Requires an API key.

Usage:
  # Dry-run (print what would be sent)
  python experiments/generate_llm_prompts.py --dry-run

  # Generate with GPT-4o
  python experiments/generate_llm_prompts.py --provider openai --model gpt-4o

  # Generate with Claude
  python experiments/generate_llm_prompts.py --provider anthropic --model claude-sonnet-4-20250514

Output:
  Patches prompts/cape_prompts.py with LLM-generated prompts.
"""

import json, sys, os, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from prompts.cape_prompts import (
    CLASS_DESCRIPTIONS, TEACHER_CLASSES, STUDENT_CLASSES,
    CAPE_LLM_GENERATED,
)

ALL_CLASSES = {
    "teacher_behavior": ["guide", "answer", "on-stage interaction",
                          "blackboard-writing", "teacher", "stand",
                          "screen", "blackboard"],
    "handrise_readwrite": ["hand-raise", "read", "write"],
    "bow_turnhead": ["bow-head", "turn-head"],
}

SYSTEM_PROMPT = """You are an expert in classroom behavior analysis.
Generate diverse, high-quality text prompts for zero-shot image classification with CLIP.
Each prompt must describe what a person doing this action looks like visually.
Be specific, vary sentence structure, and cover different viewpoints.
Return a JSON list of strings only."""


def build_user_prompt(class_name, desc):
    return f"""Generate 5 diverse text prompts for the classroom behavior class "{class_name}" (described as: "{desc}").

Guidelines:
- Focus on visual appearance (what you see in an image)
- Vary the phrasing and sentence structure
- Cover different scenarios/viewpoints
- Each prompt should be a complete sentence
- Return ONLY a JSON array of strings"""


def call_openai(api_key, model, messages):
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.7,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def call_anthropic(api_key, model, messages):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    system = messages[0]["content"]
    user = messages[1]["content"]
    resp = client.messages.create(
        model=model, system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=1024, temperature=0.7,
    )
    return resp.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str,
                        default=str(REPO_ROOT / "experiments" / "llm_generated_prompts.json"))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print(f"ERROR: Set {args.provider.upper()}_API_KEY environment variable")
        sys.exit(1)

    all_prompts = {}

    for subset_name, classes in ALL_CLASSES.items():
        for cls in classes:
            desc = CLASS_DESCRIPTIONS.get(cls, cls)
            print(f"[{subset_name}] {cls}: ", end="", flush=True)

            if args.dry_run:
                print(f"[would generate 5 prompts]")
                all_prompts[cls] = ["(dry run placeholder)"]
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(cls, desc)},
            ]

            for attempt in range(3):
                try:
                    if args.provider == "openai":
                        text = call_openai(api_key, args.model, messages)
                    else:
                        text = call_anthropic(api_key, args.model, messages)
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        for v in parsed.values():
                            if isinstance(v, list):
                                parsed = v
                                break
                    if isinstance(parsed, list) and len(parsed) >= 3:
                        all_prompts[cls] = parsed[:5]
                        print(f"✓ {len(parsed)} prompts")
                        break
                    print(f"⚠ retry (bad format)")
                except Exception as e:
                    print(f"⚠ retry ({e})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_prompts, f, indent=2)
    print(f"\n{'Dry-run: ' if args.dry_run else ''}Prompts saved to {out_path}")
    print("To use: load this file and assign to CAPE_LLM_GENERATED before experiment.")


if __name__ == "__main__":
    main()
