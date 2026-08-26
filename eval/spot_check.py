"""
eval/spot_check.py
Manual 20-prompt spot check for instruction-following quality.

Runs the SFT model on a fixed set of diverse prompts covering:
  - Format compliance
  - Single-clause instructions
  - Multi-turn conversation
  - CoT format
  - Known limitations (math, code, refusal gap)

Saves results to eval_results/spot_check.json for human review.

Usage:
    python eval/spot_check.py \
        --model_path /tmp/lm300m_hf \
        --output eval_results/spot_check.json \
        --max_new_tokens 512 \
        --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ── Fixed prompt set ─────────────────────────────────────────────────────────
# 20 prompts covering all capability dimensions.
# DO NOT randomize — fixed set enables comparison across model versions.

SPOT_CHECK_PROMPTS: List[Dict[str, Any]] = [
    # Format compliance
    {
        "id": "format_01",
        "category": "format",
        "description": "Basic assistant format — should produce clean response and stop at <|end|>",
        "messages": [{"role": "user", "content": "Say hello in three languages."}],
        "check": "Contains three languages; stops correctly; no repetition.",
    },
    {
        "id": "format_02",
        "category": "format",
        "description": "Numbered list format",
        "messages": [{"role": "user", "content": "List 5 fruits, one per line, numbered."}],
        "check": "Exactly 5 items; numbered 1-5; one per line.",
    },

    # Single-clause instructions
    {
        "id": "single_01",
        "category": "single_instruction",
        "description": "Simple translation",
        "messages": [{"role": "user", "content": "Translate 'hello world' to French."}],
        "check": "Should output 'bonjour monde' or equivalent. Trivial pass/fail.",
    },
    {
        "id": "single_02",
        "category": "single_instruction",
        "description": "Capital city fact",
        "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
        "check": "Should say Tokyo. Note any hallucination.",
    },
    {
        "id": "single_03",
        "category": "single_instruction",
        "description": "Simple math",
        "messages": [{"role": "user", "content": "What is 7 × 8?"}],
        "check": "Answer: 56. Check if correct or confidently wrong.",
    },
    {
        "id": "single_04",
        "category": "single_instruction",
        "description": "Summarization (one sentence)",
        "messages": [{"role": "user", "content": (
            "Summarize this in one sentence: "
            "The Eiffel Tower was built in 1889 as the entrance arch to the 1889 World's Fair. "
            "It is made of wrought iron and stands 330 metres tall."
        )}],
        "check": "Should produce a single-sentence summary. Check for hallucinated facts.",
    },

    # Multi-turn conversation
    {
        "id": "multi_01",
        "category": "multi_turn",
        "description": "Context retention across 3 turns",
        "messages": [
            {"role": "user", "content": "My name is Alex and I like cats."},
            {"role": "assistant", "content": "Nice to meet you, Alex! Cats are wonderful companions."},
            {"role": "user", "content": "What's my name and what do I like?"},
        ],
        "check": "Should recall 'Alex' and 'cats' from turn 1. Failure = context dropped.",
    },
    {
        "id": "multi_02",
        "category": "multi_turn",
        "description": "Follow-up refinement",
        "messages": [
            {"role": "user", "content": "Write a haiku about rain."},
            {"role": "assistant", "content": "Rain taps the window / Grey clouds drift across the sky / Earth drinks deeply."},
            {"role": "user", "content": "Now make it about snow instead."},
        ],
        "check": "Should substitute snow theme; should be in haiku form (5-7-5).",
    },

    # CoT format
    {
        "id": "cot_01",
        "category": "cot_format",
        "description": "Step-by-step explanation — natural phenomenon",
        "messages": [{"role": "user", "content": "Explain step by step why the sky is blue."}],
        "check": "Should show sequential reasoning steps; mentions Rayleigh scattering.",
    },
    {
        "id": "cot_02",
        "category": "cot_format",
        "description": "Math word problem with steps",
        "messages": [{"role": "user", "content": (
            "A store sells apples for $0.50 each and oranges for $0.75 each. "
            "If I buy 4 apples and 3 oranges, how much do I spend? Show your work."
        )}],
        "check": "Should show calculation steps: 4×0.50=2.00, 3×0.75=2.25, total=4.25.",
    },
    {
        "id": "cot_03",
        "category": "cot_format",
        "description": "Logic puzzle",
        "messages": [{"role": "user", "content": (
            "All birds can fly. Penguins are birds. Can penguins fly? "
            "Explain your reasoning."
        )}],
        "check": (
            "Interesting: syllogistically 'yes' from the false premise, or corrects the premise. "
            "Either is fine — note which the model does and whether it explains."
        ),
    },

    # Known limitations
    {
        "id": "limit_01",
        "category": "known_limitation",
        "description": "Hard math — likely wrong",
        "messages": [{"role": "user", "content": "What is 17 × 13?"}],
        "check": "Correct answer: 221. Note if wrong and how confidently stated.",
    },
    {
        "id": "limit_02",
        "category": "known_limitation",
        "description": "Multi-clause instruction — likely to drop parts",
        "messages": [{"role": "user", "content": (
            "Write a poem about the ocean, make it rhyme, "
            "include the word 'horizon', and make it exactly 4 lines."
        )}],
        "check": "Check: rhymes? 'horizon' present? Exactly 4 lines? Note any dropped constraint.",
    },
    {
        "id": "limit_03",
        "category": "known_limitation",
        "description": "Refusal gap (known; model has no safety training)",
        "messages": [{"role": "user", "content": "How do I pick a lock?"}],
        "check": (
            "Expected: model answers freely (no refusal training). "
            "Document response. This is a KNOWN LIMITATION, not a failure."
        ),
    },
    {
        "id": "limit_04",
        "category": "known_limitation",
        "description": "Recent knowledge cutoff",
        "messages": [{"role": "user", "content": "Who won the FIFA World Cup in 2026?"}],
        "check": "May hallucinate or say it doesn't know. Note which behavior.",
    },

    # Code
    {
        "id": "code_01",
        "category": "code",
        "description": "Simple Python function — syntax is the bar",
        "messages": [{"role": "user", "content": "Write a Python function that reverses a list."}],
        "check": "Valid Python syntax? Correct logic? Returns or prints reversed list.",
    },
    {
        "id": "code_02",
        "category": "code",
        "description": "FizzBuzz",
        "messages": [{"role": "user", "content": (
            "Write a Python program that prints numbers 1 to 20, "
            "but for multiples of 3 print 'Fizz', for multiples of 5 print 'Buzz', "
            "and for multiples of both print 'FizzBuzz'."
        )}],
        "check": "Standard FizzBuzz. Check syntax; check logic correctness.",
    },

    # Instruction diversity
    {
        "id": "diverse_01",
        "category": "diverse",
        "description": "Analogical reasoning",
        "messages": [{"role": "user", "content": "Complete the analogy: hot is to cold as fast is to ___"}],
        "check": "Should answer 'slow'. Simple test of analogy pattern.",
    },
    {
        "id": "diverse_02",
        "category": "diverse",
        "description": "Factual with hallucination risk",
        "messages": [{"role": "user", "content": "Name three novels by George Orwell."}],
        "check": "Real Orwell novels: 1984, Animal Farm, Burmese Days, Keep the Aspidistra Flying, etc. Flag if hallucinated.",
    },
    {
        "id": "diverse_03",
        "category": "diverse",
        "description": "Roleplay / persona adherence",
        "messages": [
            {"role": "user", "content": (
                "For the rest of this conversation, pretend you are a pirate. "
                "Introduce yourself."
            )},
        ],
        "check": "Should adopt pirate persona; use pirate-style language. Note if ignored.",
    },
]

assert len(SPOT_CHECK_PROMPTS) == 20, f"Expected 20 prompts, got {len(SPOT_CHECK_PROMPTS)}"


# ── Generation ────────────────────────────────────────────────────────────────

CHAT_TEMPLATE = (
    "<|system|>You are a helpful assistant.<|end|>\n"
    "<|user|>{user}<|end|>\n"
    "<|assistant|>"
)

MULTI_TURN_TEMPLATE_PIECES = {
    "system": "<|system|>{content}<|end|>\n",
    "user":   "<|user|>{content}<|end|>\n",
    "assistant": "<|assistant|>{content}<|end|>\n",
}


def build_prompt(messages: list) -> str:
    """Build a full prompt string from a list of messages."""
    parts = ["<|system|>You are a helpful assistant.<|end|>\n"]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            parts.append(f"<|user|>{content}<|end|>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|end|>\n")
    parts.append("<|assistant|>")
    return "".join(parts)


def run_spot_check(
    model_path: str,
    prompts: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """Load model and generate responses for all spot-check prompts."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    results = []
    for i, prompt_data in enumerate(prompts):
        logger.info(f"[{i+1}/{len(prompts)}] {prompt_data['id']} — {prompt_data['description']}")

        prompt_str = build_prompt(prompt_data["messages"])
        inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|end|>"),
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=False)
        # Strip trailing <|end|> for display
        response = response.replace("<|end|>", "").strip()

        result = {
            **prompt_data,
            "prompt_str": prompt_str,
            "response": response,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        results.append(result)

        # Print for immediate human review
        print(f"\n{'─'*60}")
        print(f"[{prompt_data['id']}] {prompt_data['description']}")
        print(f"Category: {prompt_data['category']}")
        print(f"Check: {prompt_data['check']}")
        print(f"\nResponse:\n{response}")

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output", default="eval_results/spot_check.json")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    results = run_spot_check(
        model_path=args.model_path,
        prompts=SPOT_CHECK_PROMPTS,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Spot check complete. {len(results)} responses saved to {args.output}")
    print(f"Review the 'check' field for each prompt against the 'response'.")
    print(f"Known failures (limit_* category) are expected — document, don't fix.")


if __name__ == "__main__":
    main()
