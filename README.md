# AlignScope 🔬

[![PyPI version](https://badge.fury.io/py/alignscope.svg)](https://badge.fury.io/py/alignscope)
[![Build Status](https://github.com/raghavarajunithisha-lab/AlignScopeV1/actions/workflows/publish.yml/badge.svg)](https://github.com/raghavarajunithisha-lab/AlignScopeV1/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An open-source observability SDK and real-time streaming dashboard designed to monitor and debug Multi-Agent Reinforcement Learning (MARL) systems.**

AlignScope gives machine learning engineers and researchers instant, zero-code visual feedback on complex multi-agent dynamics—including coalition formation, role specialization, reciprocity collapses, and defection events—without needing to manually parse through thousands of lines of terminal logs.

---

## 🎯 Why This Matters

Debugging multi-agent AI ecosystems is notoriously difficult. Standard MLOps tools like Weights & Biases (W&B) and MLflow are optimized for **scalar metrics** (loss, reward curves) but fall short when answering spatial or behavioral questions:

- *Why did the agent defect?*
- *When exactly did the cooperative coalition break down?*
- *Are agents actually specializing into useful roles, or just exploiting the environment?*

**AlignScope solves this by providing real-time topological graphs and event timelines** specifically engineered for multi-agent anomalies. It translates raw, high-frequency environment steps into actionable, human-readable alignment metrics.

### Key Capabilities

*   **Zero-code Integration**: One line (`alignscope.wrap(env)`) instruments any standard PettingZoo or RLlib environment.
*   **Real-Time Streaming**: High-performance WebSocket backend streams data to a 60fps D3.js and Canvas frontend.
*   **Automated Anomaly Detection**: Dynamically identifies defection, stability drops, and coalition fragmentation using continuous rolling averages.
*   **No Vendor Lock-in**: Automatically forwards intercepted telemetry back out to Weights & Biases and MLflow for persisted logging.

---

## 🏗️ Architecture Overview

AlignScope is designed as a decoupled, full-stack monitoring solution:

1.  **The SDK (Python)**: A lightweight, non-blocking telemetry tracker that normalizes raw environment step data, computes real-time alignment metrics (reciprocity, stability, convergence), and flags anomalous behavior.
2.  **The Backend (FastAPI + WebSockets)**: A robust server that manages high-frequency incoming telemetry and broadcasts it to connected clients.
3.  **The Dashboard (Vanilla JS + D3.js)**: A dark-mode, browser-based UI featuring:
    *   **Force-Directed Agent Topology**: Visualizing agent relationships, teams, and defections dynamically.
    *   **Zoomable Event Timeline**: A high-performance canvas timeline pinpointing the exact training tick where cooperation fails.

---

## 💼 Core Use Cases

**For ML Engineers:**
*   **Reduce Debugging Time**: Identify exactly when and why training instability occurs without waiting for the full epoc to finish.
*   **Track Systemic Collapse**: Monitor reciprocity and role stability drops to catch model collapse early.
*   **Offline Log Replay**: Stream huge offline CSV, JSON, NPZ, or TensorBoard training runs through the dashboard for retroactive analysis.

**For Researchers:**
*   **Study Emergent Behavior**: Measure actual role specialization (Shannon entropy) and goal convergence (Cosine similarity).
*   **Validate Alignment**: Visually confirm that your agents are forming expected coalitions in environments like SMAC (StarCraft II) or MPE.

---

## 🚀 Quick Start

### Installation

Install AlignScope directly via PyPI:

```bash
pip install alignscope
```

### Starting the Dashboard

Launch the real-time visualization server:

```bash
alignscope start
```

Open **http://localhost:8000** and watch agents interact live.


## Why AlignScope?

Training MARL agents is hard. Understanding *why* they succeed or fail is harder. Most researchers rely on reward curves and terminal logs — but those don't tell you:

- **When** did cooperation break down?
- **Which** agent defected, and why?
- **How** are coalitions forming and dissolving over time?
- **Are** agents actually specializing into useful roles?

AlignScope answers all of these questions **visually and in real time**.

### Key Advantages

| Advantage | Description |
|-----------|-------------|
| **Zero-code integration** | One line (`alignscope.wrap(env)`) to instrument any PettingZoo environment |
| **Environment agnostic** | Works with PettingZoo, SMAC, RLlib, EPyMARL, or any custom env |
| **Real-time dashboard** | Watch agents move, form coalitions, and defect as training runs |
| **Scientific precision** | Pinpoints the exact tick where cooperation fails |
| **No vendor lock-in** | Forwards metrics to W&B and MLflow automatically if installed |
| **Offline replay** | Replay saved CSV, JSON, NPZ, TensorBoard, or W&B logs into the dashboard |


## Supported MARL Environments

AlignScope is designed to work with **any** multi-agent system. Here are the environments with dedicated adapters:

### 1. PettingZoo (Cooperative & Classic Games)

PettingZoo is the most widely used MARL framework. AlignScope wraps **any** PettingZoo environment — AEC or Parallel API — with a single line.

```python
import alignscope
from pettingzoo.butterfly import knights_archers_zombies_v10

# 1. Initialize your environment
env = knights_archers_zombies_v10.env()

# 2. Add one line to wrap it with AlignScope
env = alignscope.wrap(env)  

# 3. Run your standard training loop!
env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = env.action_space(agent).sample() if not (term or trunc) else None
    env.step(action)
```

*(See `examples/` for templates using RLlib, SMAC, and offline log replay.)*

---

## 🤝 Contributing & Support

We welcome contributions! Whether it's adding a new environment adapter, optimizing the frontend canvas renderer, or designing new alignment metrics.

For support, issues, or professional inquiries, please [open an issue](https://github.com/raghavarajunithisha-lab/AlignScopeV1/issues) or reach out directly:
📧 **nithisha2201@gmail.com**

*Designed and maintained by Nithisha.*
