#!/usr/bin/env python3
"""Generate diverse LLM fiction corpus for slop classifier training.

Sends writing prompts to multiple models via OpenRouter in parallel,
saves output organized by model.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"
AUTH_FILE = Path.home() / ".local/share/opencode/auth.json"
OUTPUT_DIR = Path(__file__).parent / "corpus"

MODELS = [
    "qwen/qwen3-30b-a3b",
    "mistralai/mistral-small-creative",
    "meta-llama/llama-4-scout",
    "deepseek/deepseek-v3.2",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "nousresearch/hermes-4-70b",
    "stepfun/step-3.5-flash",
    "qwen/qwen3.5-flash-02-23",
]

# Diverse prompts across genres and scene types
PROMPTS = [
    # Literary / introspective
    "Write the opening chapter (2000 words) of a literary novel about a cartographer mapping the last unmapped territory on Earth. Third person limited, present tense. The character discovers something that challenges their understanding of the landscape. Include sensory detail, internal monologue, and at least two other characters with dialogue.",
    "Write a chapter (2000 words) from a literary novel about a woman returning to her childhood home after her mother's death. She discovers letters that reframe her understanding of her parents' marriage. Third person limited, past tense. Focus on the physical space of the house and how memory attaches to objects.",
    "Write a chapter (2000 words) about a translator working on a dead language who realizes the text describes events that haven't happened yet. Literary fiction, third person limited. Include scenes of close reading, a conversation with a skeptical colleague, and the moment of realization.",

    # Sci-fi
    "Write a chapter (2000 words) of hard science fiction about an engineer on a generation ship who discovers the navigation system has been subtly altered. First person, past tense. Include technical detail about the ship's systems, a tense conversation with the ship's AI, and the character's growing paranoia.",
    "Write a chapter (2000 words) about first contact told from the perspective of a linguist trying to decode an alien signal. Third person limited, present tense. The aliens communicate through mathematical structures that map to emotions. Include the process of decoding and a breakthrough moment.",
    "Write a chapter (2000 words) set in a near-future city where a climate refugee navigates a new social hierarchy. Third person close, past tense. Focus on the physical environment, bureaucratic encounters, and a moment of unexpected kindness from a stranger.",

    # Fantasy
    "Write a chapter (2000 words) of low fantasy about a healer in a medieval city during a plague. She uses real herbal medicine but is accused of witchcraft. Third person limited, past tense. Include a patient interaction, a confrontation with a priest, and a private moment of doubt.",
    "Write a chapter (2000 words) about a retired soldier who tends a garden on the edge of a forest where the trees move. Literary fantasy, third person close, past tense. The forest is encroaching. Include gardening detail, a visit from a neighbor, and the character's relationship with violence.",
    "Write a chapter (2000 words) of secondary world fantasy about a scribe who discovers that the historical records she copies have been systematically altered. Third person limited. Include the moment of discovery, her attempt to verify it, and the political danger of knowing.",

    # Thriller / suspense
    "Write a chapter (2000 words) of a psychological thriller about a forensic accountant who finds evidence that her firm is laundering money. First person, present tense. Include the discovery scene, a normal-seeming meeting that takes on new meaning, and the decision of whether to act.",
    "Write a chapter (2000 words) about a night shift nurse who notices a pattern in patient deaths that nobody else has caught. Third person limited, past tense. Include a detailed medical scene, a conversation with a dismissive supervisor, and the nurse's growing certainty.",

    # Character study / quiet drama
    "Write a chapter (2000 words) about two estranged siblings meeting at their father's funeral. Third person omniscient, past tense. Alternate between their perspectives. Include the physical details of the funeral, unspoken tension, and a single moment of genuine connection.",
    "Write a chapter (2000 words) about an aging chess grandmaster playing his last tournament. Third person limited, present tense. Interweave the current game with memories of his peak. Include the physicality of the chess pieces, his opponent's mannerisms, and the crowd.",

    # Horror / gothic
    "Write a chapter (2000 words) of literary horror about a marine biologist studying a deep-sea trench who begins receiving structured signals from below. Third person limited, past tense. The horror is slow and atmospheric, not violent. Include scientific process, isolation, and growing unease.",
    "Write a chapter (2000 words) of southern gothic about a woman who inherits a house with a locked room. Third person close, past tense. The house has a history the town won't discuss. Include interactions with suspicious neighbors, the physical decay of the house, and what she hears through the locked door.",

    # Romance / relationship
    "Write a chapter (2000 words) about two rival academics who are forced to collaborate on a research project. Third person alternating, past tense. Include intellectual sparring that masks attraction, a late-night breakthrough in the lab, and the moment one of them drops their guard.",
    "Write a chapter (2000 words) about a war correspondent returning home to a partner who has built a life without them. Third person limited from the correspondent's POV, present tense. Focus on the gap between the world they left and the one they returned to.",

    # Historical
    "Write a chapter (2000 words) set in 1920s Istanbul about a calligrapher whose art form is being replaced by the Latin alphabet reform. Third person limited, past tense. Include the physical act of calligraphy, a political argument with a reformist friend, and the character's private grief.",
    "Write a chapter (2000 words) about a female astronomer in 17th century Italy who must publish her discoveries under a male colleague's name. Third person close, past tense. Include an observation session, the collaboration/exploitation dynamic, and a discovery that changes everything.",

    # Post-apocalyptic
    "Write a chapter (2000 words) set twenty years after a solar EMP about a community that has rebuilt around a working hydroelectric dam. Third person limited, past tense. A stranger arrives claiming to represent a larger settlement. Include the community's daily rhythms, the political debate about contact, and sensory detail of the rebuilt world.",

    # Magical realism
    "Write a chapter (2000 words) of magical realism about a baker whose bread reveals the eater's buried memories. Third person limited, present tense. A customer comes in who the baker recognizes but shouldn't. Include the baking process, the shop's atmosphere, and the moment the bread does its work.",

    # Coming of age
    "Write a chapter (2000 words) about a teenager in a small fishing town who discovers their grandfather's secret life as a smuggler. Third person limited, past tense. Include the physical world of the harbor, a conversation with a local who knew the grandfather, and the teen's shifting understanding of family loyalty.",

    # Dystopia
    "Write a chapter (2000 words) of dystopian fiction about a memory editor — someone whose job is to remove traumatic memories from citizens. Third person limited, present tense. A client's memory reveals something about the regime that the editor was not supposed to see. Include the technical process, the ethical weight, and the decision point.",

    # War fiction
    "Write a chapter (2000 words) about a combat medic during a ceasefire, treating wounded from both sides. Third person limited, past tense. Include triage decisions, a conversation with an enemy soldier, and the medic's internal calculus about who to save.",

    # Noir / crime
    "Write a chapter (2000 words) of modern noir about a private investigator hired to find a missing person who, it turns out, doesn't want to be found. First person, past tense. Include legwork (interviewing witnesses, checking records), a dead end that becomes a lead, and the PI's personal code clashing with the case.",
]

def load_api_key() -> str:
    with open(AUTH_FILE) as f:
        return json.load(f)["openrouter"]["key"]


async def generate_one(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    prompt_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate one chapter from one model."""
    async with semaphore:
        model_short = model.split("/")[-1]
        print(f"  [{model_short}] prompt {prompt_idx}...", flush=True)

        try:
            resp = await client.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a fiction writer. Write exactly what is asked. "
                                "No preamble, no commentary, no meta-discussion. "
                                "Start directly with the prose."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.9,
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            cost_str = data.get("usage", {}).get("cost", "?")

            return {
                "model": model,
                "prompt_idx": prompt_idx,
                "text": text,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cost": cost_str,
                "error": None,
            }
        except Exception as e:
            print(f"  [{model_short}] prompt {prompt_idx} FAILED: {e}", flush=True)
            return {
                "model": model,
                "prompt_idx": prompt_idx,
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cost": 0,
                "error": str(e),
            }


async def main():
    api_key = load_api_key()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build all tasks: every model x every prompt
    tasks = []
    for model in MODELS:
        for pi, prompt in enumerate(PROMPTS):
            tasks.append((model, prompt, pi))

    print(f"Generating {len(tasks)} chapters ({len(MODELS)} models x {len(PROMPTS)} prompts)")
    print(f"Models: {', '.join(m.split('/')[-1] for m in MODELS)}")

    # 15 concurrent requests to avoid rate limits
    semaphore = asyncio.Semaphore(15)
    results = []

    async with httpx.AsyncClient() as client:
        coros = [
            generate_one(client, api_key, model, prompt, pi, semaphore)
            for model, prompt, pi in tasks
        ]
        results = await asyncio.gather(*coros)

    # Save per-model
    total_tokens = 0
    total_cost = 0
    errors = 0
    for model in MODELS:
        model_short = model.split("/")[-1]
        model_dir = OUTPUT_DIR / model_short
        model_dir.mkdir(exist_ok=True)

        model_results = [r for r in results if r["model"] == model]
        for r in model_results:
            if r["error"]:
                errors += 1
                continue
            outfile = model_dir / f"prompt_{r['prompt_idx']:02d}.md"
            outfile.write_text(r["text"])
            total_tokens += r["output_tokens"]
            try:
                total_cost += float(r["cost"])
            except (ValueError, TypeError):
                pass

    # Summary
    success = len(results) - errors
    print(f"\nDone: {success}/{len(results)} successful, {errors} errors")
    print(f"Total output tokens: {total_tokens:,}")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(f"Output: {OUTPUT_DIR}/")

    # Save manifest
    manifest = {
        "models": MODELS,
        "prompts": PROMPTS,
        "results": [
            {k: v for k, v in r.items() if k != "text"}
            for r in results
        ],
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
