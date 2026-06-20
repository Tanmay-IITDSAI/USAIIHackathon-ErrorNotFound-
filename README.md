<div align="center">

# 🧠 Second Brain: Cognitive Scaffold Engine v3.0

### *Zero-to-One Builder — 100% Free · GPU-Accelerated · Voice-Enabled · Benchmarked*

**Turns a vague ambition into a structured, narrated 30/60/90-day execution plan — grounded in peer-reviewed behavioral science, not vibes.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/second-brain-cognitive-scaffold/blob/main/second_brain_huggingfacev3_Tanmay.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Qwen2.5--7B-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Hackathon%20Submission-orange)
![GPU](https://img.shields.io/badge/GPU-Colab%20T4%20Free%20Tier-76B900?logo=nvidia&logoColor=white)

**USAII Global AI Hackathon 2026 · Challenge Brief 3 · Undergraduate Track**
**Team:** Error: Skill Not Found! — Tanmay Kumar Shrivastava, Mansu Khute

</div>

---

> *"How might we use AI to help people think more clearly, understand tradeoffs, and move from uncertainty to meaningful action?"*

## 📖 Table of Contents

- [What This Is](#-what-this-is)
- [Why It's Different](#-why-its-different-v1v2--v30)
- [System Architecture](#-system-architecture)
- [Model Stack](#-model-stack-100-free-colab-t4-optimized)
- [Research Foundation](#-research-foundation)
- [Quickstart](#-quickstart)
- [Usage](#-usage)
- [Example Output](#-example-output)
- [Project Structure](#-project-structure)
- [Evaluation Benchmark](#-evaluation-benchmark)
- [Responsible AI](#-responsible-ai)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [References](#-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Team](#-team)

---

## 🎯 What This Is

A college student types one vague sentence about their future — *"I want to work in AI but I'm a mechanical engineering student with no CS background"* — and **Second Brain** runs it through a 5-stage cognitive pipeline that produces:

1. A genuine clarification of their goal, constraints, hidden motivation, and #1 skill gap
2. The 3 most dangerous hidden assumptions in their plan, risk-scored and paired with a 7-day test to validate or kill each one
3. A confidence-scored 30/60/90-day milestone plan built only from free resources
4. One concrete, zero-cost action they can take in the next 45 minutes
5. An explicit list of what the AI *cannot* know about them — because the model should not pretend to replace human judgment

The entire plan is **narrated aloud** stage-by-stage, visualized as a readiness radar chart, exported to Markdown + MP3, and automatically **benchmarked** for specificity, relevance, and template leakage — all running on a **free Google Colab T4 GPU**, with no API keys and no gated models.

## 🆚 Why It's Different (v1/v2 → v3.0)

Earlier iterations of this project used encoder-decoder models (T5 family) and ran into a specific, well-understood failure mode: **template echoing**.

| Problem in v1 / v2 | Root Cause | v3.0 Fix |
|---|---|---|
| Milestones literally said `[specific action]` | T5-style models echo template brackets given in the prompt instead of filling them in | Switched to **Qwen2.5-7B-Instruct** (decoder-only) — never echoes templates |
| First step said `I will [[start]] at [[time]]` | `flan-t5-xl` copies format strings rather than generating real content | Chat-template generation produces real, complete sentences |
| Assumptions were nonsensical or generic | 248M–3B encoder-decoder models can't sustain multi-step reasoning | 7.6B decoder-only model with materially stronger reasoning |
| `total_mem` `AttributeError` | Wrong PyTorch attribute name | Fixed to `total_memory` |
| No way to know if outputs were actually good | No evaluation step existed | Added an automated benchmark scoring specificity, relevance, diversity, and template leakage |

**The architectural insight:** T5/Flan-T5 models are trained on text-to-text mapping tasks and tend to literally reproduce template formats given in a prompt. Decoder-only, instruction-tuned chat models like Qwen2.5 are trained on multi-turn conversations and understand that template examples are meant to be filled in, not echoed back. This single change is the primary reason v3.0 outputs are dramatically more specific and usable than v1/v2.

## 🏗️ System Architecture

```
USER INPUT  (vague idea, natural language)
      │
      ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: IDEA CLARIFIER                     │
│  Model: Qwen2.5-7B-Instruct (4-bit)          │
│  Framework: Cognitive Load Theory             │
│  Output: Goal · Constraints · Hidden Why      │
│  🗣️ Voice: Narrates clarification              │
└──────────────────────┬───────────────────────┘
                        ▼
┌─────────────────────────────────────────────┐
│  STAGE 2: ASSUMPTION MINER                   │
│  Model: Qwen2.5-7B + bart-large-mnli         │
│  Framework: Assumption-Based Planning         │
│  Output: 3 Hidden Risks + NLI Scores          │
└──────────────────────┬───────────────────────┘
                        ▼
┌─────────────────────────────────────────────┐
│  STAGE 3: MILESTONE BUILDER                  │
│  Model: Qwen2.5-7B + bart-large-mnli         │
│  Output: 30/60/90-day plan + Confidence       │
│  📊 Visual: Readiness radar chart              │
└──────────────────────┬───────────────────────┘
                        ▼
┌─────────────────────────────────────────────┐
│  STAGE 4: FIRST STEP FORGE                   │
│  Model: Qwen2.5-7B-Instruct                  │
│  Framework: Implementation Intentions         │
│  Output: Specific action for today            │
│  🗣️ Voice: Motivational first-step coach       │
└──────────────────────┬───────────────────────┘
                        ▼
┌─────────────────────────────────────────────┐
│  STAGE 5: RESPONSIBLE AI + EVALUATION        │
│  Explicit uncertainty · Human handoff         │
│  Automated quality scoring · 🗣️ Full narration │
└─────────────────────────────────────────────┘
```

Each stage is its own Python function (`stage1_clarify`, `stage2_assumptions`, `stage3_milestones`, `stage4_first_step`, `stage5_responsible_ai`), orchestrated end-to-end by `run_second_brain()`.

## 🤖 Model Stack (100% Free, Colab T4 Optimized)

| Model | Params | VRAM (4-bit) | Role | Why this model? |
|---|---|---|---|---|
| [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | 7.6B | ~5 GB | Reasoning, planning, clarification | Decoder-only instruction-tuned model. Unlike T5 (encoder-decoder), it generates fluent, specific, contextual text instead of echoing templates. Top-ranked on MT-Bench and AlpacaEval. |
| [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) | 407M | ~1.6 GB | Zero-shot risk & confidence classification | NLI-based confidence scoring without any labelled data. |
| [`gTTS`](https://gtts.readthedocs.io/) (Google TTS) | — | 0 | Voice narration of the plan | Free Google Translate TTS. Reads the entire plan aloud, stage by stage. |

**Total VRAM: ~7 GB on a T4 (16 GB available). Free forever. No API key. No gated repos.**

> On CPU-only environments, the pipeline automatically falls back to `Qwen/Qwen2.5-1.5B-Instruct` so the notebook still runs — though a free Colab T4 is strongly recommended.

## 🔬 Research Foundation

| Framework | Source | Applied In |
|---|---|---|
| **Cognitive Load Theory** | Sweller (1988), *Cognitive Science* | Stage 1 — Idea Clarifier |
| **Assumption-Based Planning** | Dewar, Dutton & Reardon (2002), *McKinsey Quarterly* | Stage 2 — Assumption Miner |
| **Implementation Intentions** | Gollwitzer (1999), *American Psychologist* | Stage 4 — First Step Forge |

## 🚀 Quickstart

### Option A — Google Colab (recommended, zero setup)

1. Click **Open in Colab** above.
2. Set the runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.
3. Run all cells top to bottom (`Runtime → Run all`). First run downloads ~5 GB of model weights; subsequent runs are instant thanks to Colab's cache.

### Option B — Local / self-hosted GPU

```bash
# Clone the repo
git clone https://github.com/<your-username>/second-brain-cognitive-scaffold.git
cd second-brain-cognitive-scaffold

# Create an environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch
jupyter notebook second_brain_huggingfacev3_Tanmay.ipynb
```

**Requirements:** Python 3.10+, an NVIDIA GPU with ≥8 GB VRAM (4-bit quantization), CUDA-enabled PyTorch. CPU-only machines work but fall back to the 1.5B model and run noticeably slower.

## 🛠️ Usage

The notebook is organized into runnable cells:

| Cell | What it does |
|---|---|
| 1 — Install Dependencies | Installs `transformers`, `torch`, `accelerate`, `bitsandbytes`, `gTTS`, `matplotlib` |
| 2 — Load Models | Loads Qwen2.5-7B-Instruct in 4-bit NF4 quantization + bart-large-mnli |
| 3 — Core Utilities | Generation helper, zero-shot classifier, TTS engine, radar chart generator, HTML display helpers |
| 4 — Stage Functions | The 5 cognitive scaffold stages (clarify → assumptions → milestones → first step → responsible AI) |
| 5 — Pipeline Engine | `run_second_brain(user_idea, enable_voice=True)` — orchestrates all 5 stages end-to-end |
| 6 — Demo Run | Runs the pipeline on a sample idea out of the box |
| 7 — Interactive Mode | Prompts you to type your own idea via `input()` |
| 8 — Export Plan | Exports the full plan as a timestamped `.md` file + narrated `.mp3` |
| 9 — Evaluation Benchmark | Runs the automated quality benchmark across 3 diverse test ideas |
| 10 — Devpost Fields | Pre-formatted submission text (elevator pitch, architecture summary, etc.) |

**Minimal usage example:**

```python
user_idea = "I want to freelance as a UI/UX designer while finishing my degree"
result = run_second_brain(user_idea, enable_voice=True)
export_plan(result, user_idea)   # writes execution_plan_<timestamp>.md and .mp3
```

## 📋 Example Output

Running the demo idea — *"I want to work in AI after graduation but I'm a mechanical engineering student with no CS background"* — produces a structured plan like:

```markdown
## Stage 1: Idea Clarification
**Core Goal:** ...
**Real Constraints:** ...
**Hidden Motivation:** ...
**Critical Skill Gap:** ...

## Stage 2: Hidden Assumptions
### Assumption 1 — Risk: High (87%)
**Assumption:** They assume that ...
**Test:** ...

## Stage 3: 30/60/90-Day Plan
### Days 1–30 — FOUNDATION
- M1 [High 91%]: ...
- M2 [Medium 68%]: ...
- M3 [High 84%]: ...

## Stage 4: First Step
> Tonight at 7pm, I will ...

**Avoid:** ...
**Human decision:** Only you can decide ...

## Responsible AI
**Uncertainty:** High — career path requires experimentation and pivoting
**AI cannot know:** ...
```

In the notebook, each stage also renders as a styled HTML card, with a live readiness radar chart and an inline audio player narrating that stage aloud.

## 📁 Project Structure

```
second-brain-cognitive-scaffold/
├── second_brain_huggingfacev3_Tanmay.ipynb   # The full pipeline (single-notebook project)
├── README.md                                  # You are here
├── requirements.txt                           # Pinned Python dependencies
├── LICENSE                                    # MIT License
├── .gitignore                                 # Ignores model caches, exports, env files
└── outputs/                                    # (generated at runtime)
    ├── execution_plan_<timestamp>.md           # Exported plan
    └── execution_plan_<timestamp>.mp3          # Narrated audio version
```

## 📊 Evaluation Benchmark

Cell 9 runs an automated, model-graded benchmark across 3 diverse test ideas (AI career switch, freelance UI/UX, biology → health-tech), scoring Stage 1 output on four dimensions using `bart-large-mnli` zero-shot NLI plus rule-based checks:

| Metric | How it's measured |
|---|---|
| **Specificity** | NLI classification: does the text read as specific/actionable vs. vague/generic? |
| **Relevance** | NLI classification: is the response relevant to the student's stated idea? |
| **No Template Leak** | Regex scan for residual `[bracket]` / `[[bracket]]` placeholders — the exact failure mode v3.0 was built to eliminate |
| **Detail Depth** | Word count, capped and normalized to a 0–100 scale |

This turns "does the output feel better?" into a repeatable, numeric check — useful both for judging hackathon submissions and for catching regressions if the prompts or model are changed later.

## 🛡️ Responsible AI

This project treats the AI as a **thinking partner, not a decision-maker**:

- **Explicit uncertainty:** every plan is tagged with a Low/Medium/High path-uncertainty score via zero-shot NLI, not asserted as fact.
- **Stated limitations:** Stage 5 explicitly lists personal, relational, and values-based factors the AI cannot know about the user and that materially change the right answer.
- **Human handoff:** Stage 4 always ends with a values-based question the user — not the model — must answer before committing.
- **No prescriptions:** every exported plan ends with the line *"This plan is a starting hypothesis, not a prescription."*

## 🧰 Tech Stack

`Python` · `Jupyter / Google Colab` · `🤗 Transformers` · `PyTorch` · `bitsandbytes` (4-bit NF4 quantization) · `gTTS` · `matplotlib` · `NumPy`

**Hardware:** Google Colab T4 GPU, free tier (16 GB VRAM) — full 5-stage pipeline runs in ~90–120 seconds.

## 🗺️ Roadmap

- [ ] Swap `input()`-based interactive mode for a Gradio/Streamlit front end
- [ ] Add persistent plan history / multi-session comparison
- [ ] Support additional languages in TTS narration
- [ ] Expand the benchmark to cover Stages 2–4, not just Stage 1
- [ ] Optional local LLM swap (e.g., Llama 3, Mistral) behind the same `generate()` interface

## 📚 References

**Behavioral Science**
- Sweller, J. (1988). Cognitive load during problem solving. *Cognitive Science, 12*(2), 257–285.
- Gollwitzer, P. M. (1999). Implementation intentions. *American Psychologist, 54*(7), 493–503.
- Gollwitzer, P. M., & Sheeran, P. (2006). Meta-analysis of implementation intention effects. *European Review of Social Psychology, 38*, 69–119. *(N = 8,461)*
- Dewar, C., Dutton, J., & Reardon, T. (2002). Assumption-Based Planning. *McKinsey Quarterly.*

**AI Models**
- Qwen Team (2024). Qwen2.5: A Party of Foundation Models. *arXiv:2412.15115.*
- Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. *ACL 2020.*
- Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023.*

## 🤝 Contributing

Contributions are welcome. To propose a change:

1. Fork the repo and create a branch: `git checkout -b feature/your-feature`
2. Make your changes (and re-run the notebook to confirm it still executes cleanly top to bottom)
3. Commit with a clear message and open a pull request describing what changed and why

Bug reports and prompt-quality issues (e.g., a recurrence of template leakage) are especially appreciated — open an issue with the offending idea/prompt and the raw model output.

## 📄 License

This project is released under the [MIT License](LICENSE). Model weights are governed by their own licenses: Qwen2.5-7B-Instruct (Apache 2.0) and bart-large-mnli (MIT) — both free for commercial and research use.

## 👥 Team

**Team Name:** Error: Skill Not Found!
**Members:** Tanmay Kumar Shrivastava, Mansu Khute
**Event:** USAII Global AI Hackathon 2026 — Challenge Brief 3, Undergraduate Track

---

<div align="center">

*Second Brain v3.0: from confusion → clarity → action → voice.*
*Free. Benchmarked. Grounded in science.*

</div>
