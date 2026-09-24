# Hillock

**A lightweight, 100% local memory engine built for edge hardware.**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![VRAM](https://img.shields.io/badge/VRAM-%3C1.2GB-brightgreen)
![Status](https://img.shields.io/badge/status-v0.6.1-orange)

Traditional local RAG is heavy. Running dense vector databases and using 8B+ generative LLMs just to parse documents burns VRAM, chokes mid-range GPUs, and still hallucinates when asked about things it doesn't know.

Hillock was built to solve this. It replaces bloated vector databases and token-hungry extraction passes with a lightweight, three-tier architecture: a relational SQLite Knowledge Graph, Hebbian synaptic memory, and Hyperdimensional Computing (HDC). 

It extracts facts, blocks unanswerable questions mathematically before they ever reach the LLM, and runs entirely in under 1.2 GB of VRAM.

### Why use Hillock?
* **Zero Hallucinations:** A hard mathematical gate blocks questions it doesn't know the answer to. It refuses honestly instead of guessing.
* **Extremely Fast Ingestion:** It uses tensor-based classification instead of an LLM to read documents. It can ingest a 30-sentence document in about 5 seconds.
* **Runs on a Potato:** The entire pipeline fits in <1.2 GB VRAM and can even run CPU-only if needed.
* **API Ready (New in v0.6.x):** Hillock now includes an OpenAI-compatible API server. You can plug its hallucination-free memory directly into UIs like Open-WebUI or Obsidian.

---

## 🚀 Quick Start

**Prerequisites:** Python 3.10+ and [Ollama](https://ollama.com/) running locally.

### Option A: 1-Click Launch (Recommended)
The launcher scripts create the virtual environment, install dependencies, and start the console automatically.

**Windows:**
```bat
run.bat
```

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual Setup
```bash
git clone https://github.com/roandejager/Hillock.git
cd Hillock

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install the engine and its dependencies
pip install -e .

# Download the required English language model
python -m spacy download en_core_web_sm
```

---

## 🕹️ How to Use Hillock

Because Hillock is decoupled, you can use it in three different ways depending on your needs.

### 1. The API Server (For Custom UIs)
You can run Hillock in the background and connect it to your favorite AI interface (like Open-WebUI or AnythingLLM).
```bash
python api.py
```
This starts a local server at `http://localhost:8000`. Just go into your UI's settings, set the OpenAI API Base URL to `http://localhost:8000/v1`, and chat normally.

### 2. The Terminal Console
If you prefer the classic hacker aesthetic, you can chat with your documents directly in the terminal.
```bash
python main.py
```
**Helpful CLI Commands:**
* `/ingest [file.txt or .pdf]` - Feed a document into the memory engine.
* `/model [name]` - Switch your local Ollama model on the fly.
* `/mode [strict | balanced | conversational]` - Change how the assistant talks.
* `/inspect [entity]` - Look under the hood at exactly what the engine knows about a topic.

### 3. The Python Library (For Developers)
You can import Hillock directly into your own Python applications.
```python
from engine import IntegratedHillock

# Initialize the memory engine
my_brain = IntegratedHillock("my_database.db")

# Query it programmatically
answer, primed_nodes, hdc_traces, mode = my_brain.execute_chat_turn("What did Alan Turing crack?")
print(answer)
```

---

## ✅ Verification Suite

Hillock includes a standalone, GPU-free test suite that checks the core mathematical invariants, database locks, and gating logic. It is safe to wire into CI pipelines without a GPU runner.

```bash
python verify_hillock.py
```

---

## 🗺️ Roadmap to v1.0

The full, detailed roadmap is tracked in **[Issue #1: The Path to v1.0](https://github.com/roandejager/Hillock/issues/1)**. 

Our immediate next steps focus on optimizing the engine for Small Language Models (SLMs) to create a conversational, proactive agent, followed by extreme hardware optimizations like bit-packed hypervectors to push CPU speeds even higher.

---

## ⚖️ Licensing & Contributions

Licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

To keep the project open-source while preserving the option for future commercial dual-licensing, contributors must sign a standard **Contributor License Agreement (CLA)** via `cla-assistant.io` when opening a PR. See `CONTRIBUTING.md` and `CLA.md`.

## 📬 Contact & Collaboration

Hillock is an active research project by **Roan de Jager**. 

If you are interested in custom memory integrations, consulting on edge-AI systems, or commercial licensing, feel free to reach out directly via email at **[contact.roandejager@gmail.com](mailto:contact.roandejager@gmail.com)** or open a discussion on GitHub.
