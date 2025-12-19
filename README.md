# TinyLLM

> **What if each neuron in a neural network was already intelligent?**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│    Traditional Neural Network          TinyLLM Neural Network               │
│    ═══════════════════════════         ═══════════════════════              │
│                                                                             │
│         ○ ○ ○ ○ ○                           🧠 🧠 🧠                        │
│        ╱│╲│╱│╲│╱│╲                         ╱  │  ╲                          │
│       ○ ○ ○ ○ ○ ○ ○                      🧠   🧠   🧠                       │
│        ╲│╱│╲│╱│╲│╱                        ╲   │   ╱                         │
│         ○ ○ ○ ○ ○                           🧠 🧠                           │
│                                              │                               │
│    Millions of simple neurons              🧠                               │
│    → Emergent intelligence              Dozens of intelligent neurons       │
│                                         → Emergent superintelligence        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](docs/CONTRIBUTING.md)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-blueviolet)](https://ollama.ai)

---

## The Concept

TinyLLM treats small language models (≤3B parameters) as **intelligent neurons** in a larger cognitive architecture:

| Component | Traditional NN | TinyLLM |
|-----------|---------------|---------|
| **Neuron** | Simple activation function | Entire small LLM |
| **Weights** | Learned parameters | Routing probabilities + prompts |
| **Learning** | Backpropagation | LLM-as-judge + recursive expansion |
| **Inference** | Forward pass | Multi-step reasoning with tools |

### Key Innovations

- **Recursive Self-Improvement**: When a node fails, it automatically expands into a router + multiple specialist strategies
- **Tool-Augmented Neurons**: Models can call calculators, code executors, and search—shifting computation off the LLM
- **Gamified Training**: Nodes earn XP, level up, and compete on leaderboards
- **100% Local**: Runs entirely on consumer hardware via Ollama

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- At least one small model: `ollama pull qwen2.5:3b`

### Installation

```bash
# Clone the repository
git clone https://github.com/ndjstn/tinyllm.git
cd tinyllm

# Install dependencies
pip install -e .

# Pull recommended models
ollama pull qwen2.5:0.5b   # Router (tiny, fast)
ollama pull qwen2.5:3b     # General specialist
ollama pull granite-code:3b # Code specialist

# Verify installation
tinyllm doctor
```

### First Run

```bash
# Initialize default configuration
tinyllm init

# Run a simple query
tinyllm run "What is 2 + 2?"

# Run with trace output
tinyllm run --trace "Write a Python function to check if a number is prime"

# Interactive mode
tinyllm chat
```

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            USER QUERY                                       │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         ENTRY NODE                                    │  │
│  │                    (Input validation)                                 │  │
│  └───────────────────────────┬──────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      TASK ROUTER                                      │  │
│  │              qwen2.5:0.5b (classification)                            │  │
│  │         "Is this code? math? factual? creative?"                      │  │
│  └───────────────────────────┬──────────────────────────────────────────┘  │
│               ┌──────────────┼──────────────┐                               │
│               ▼              ▼              ▼                               │
│  ┌────────────────┐ ┌───────────────┐ ┌────────────────┐                   │
│  │  CODE SPECIALIST│ │MATH SPECIALIST│ │GENERAL SPECIALIST│                │
│  │  granite-code:3b│ │   phi3:mini   │ │   qwen2.5:3b    │                 │
│  │  + code_executor│ │  + calculator │ │                 │                 │
│  └────────┬───────┘ └───────┬───────┘ └────────┬────────┘                  │
│           │                 │                   │                           │
│           └─────────────────┴───────────────────┘                           │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       QUALITY GATE                                    │  │
│  │           Rule-based checks + optional LLM judge                      │  │
│  │                 Pass → Exit | Fail → Retry/Expand                     │  │
│  └───────────────────────────┬──────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         RESPONSE                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### How Recursive Expansion Works

When a node consistently fails:

```
BEFORE (failing):                    AFTER (expanded):

┌─────────────┐                      ┌─────────────┐
│ math_solver │ ────(fails 40%)────► │ math_router │
└─────────────┘                      └──────┬──────┘
                                       ┌────┴────┐
                                       ▼         ▼
                                ┌──────────┐ ┌──────────┐
                                │arithmetic│ │ algebra  │
                                │ solver   │ │ solver   │
                                └──────────┘ └──────────┘
```

The system learns which sub-strategies work for which types of math problems.

---

## Model Tiers

TinyLLM uses a tiered model architecture:

| Tier | Purpose | Models | VRAM |
|------|---------|--------|------|
| **T0** | Routers | qwen2.5:0.5b, tinyllama | ~500MB |
| **T1** | Specialists | granite-code:3b, qwen2.5:3b, phi3:mini | 2-3GB |
| **T2** | Workers | qwen3:8b | 5-6GB |
| **T3** | Judges | qwen3:14b, gpt-oss:20b | 10-15GB |

Small models handle routing (fast, cheap). Larger models only used for judging/grading.

---

## Hardware Requirements

**Minimum:**
- 16GB RAM
- 8GB VRAM (single GPU)
- 50GB disk

**Recommended (our setup):**
- 128GB RAM
- 2× RTX 3060 (24GB VRAM total)
- AMD Ryzen 7 3700X

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | Deep dive into system design |
| [Contributing](docs/CONTRIBUTING.md) | How to contribute |
| [Roadmap](docs/ROADMAP.md) | What's planned |
| [Specifications](docs/specs/) | Detailed component specs |

### Concept Guides

- [Neural Network of LLMs](docs/concepts/neural-network-of-llms.md)
- [Recursive Expansion](docs/concepts/recursive-expansion.md)
- [Self-Improvement Loop](docs/concepts/self-improvement-loop.md)
- [Gamification](docs/concepts/gamification.md)

---

## Contributing

We welcome contributions! TinyLLM is designed for parallel development:

```bash
# Find issues you can work on
gh issue list --label "good-first-issue"
gh issue list --label "help-wanted"

# Each issue is atomic and self-contained
# Pick one, implement it, submit a PR
```

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

### Contribution Areas

| Area | Skills Needed | Examples |
|------|---------------|----------|
| 🐍 Core | Python, async | Implement nodes, executor |
| 📝 Prompts | Prompt engineering | Write/improve prompts |
| 🧪 Testing | Python, pytest | Write test cases |
| 📖 Docs | Technical writing | Improve documentation |
| 🔧 Tools | Python | Implement calculator, code executor |
| 📊 Research | ML knowledge | Benchmarking, analysis |

---

## Roadmap

### Phase 0: Foundation ✨ **Current**
- [ ] Config loading system
- [ ] Pydantic models
- [ ] Ollama async client
- [ ] Basic message types

### Phase 1: Core Engine
- [ ] Node base class
- [ ] Graph structure
- [ ] Executor

### Phase 2: First Tools
- [ ] Calculator
- [ ] Code executor
- [ ] Web search

### Phase 3: Routing & Specialists
- [ ] Router node
- [ ] Model node
- [ ] Initial prompts

### Phase 4: Grading System
- [ ] LLM-as-judge
- [ ] Metrics tracking
- [ ] Failure analysis

### Phase 5: Self-Improvement
- [ ] Expansion triggers
- [ ] Graph mutations
- [ ] Pruning

See [ROADMAP.md](docs/ROADMAP.md) for detailed timeline.

---

## Philosophy

> "The best way to predict the future is to invent it." — Alan Kay

We believe:

1. **Small models are underrated**: With the right orchestration, small models can match large ones
2. **Tools beat parameters**: A 3B model with a calculator beats a 70B model doing mental math
3. **Self-improvement is possible**: Systems can learn from their mistakes without human intervention
4. **Local is the future**: Privacy, cost, and latency all favor local inference

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [Ollama](https://ollama.ai) - Local LLM inference
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph orchestration
- [Pydantic](https://pydantic.dev) - Data validation

---

<p align="center">
  <strong>⭐ Star us on GitHub if you find this interesting! ⭐</strong>
</p>
