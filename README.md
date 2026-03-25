# AlignScope

**Real-time alignment observability for Multi-Agent Reinforcement Learning.**

AlignScope is an open-source SDK and live dashboard that lets MARL researchers **see** what their agents are doing — in real time. Drop it into any multi-agent training pipeline and get instant visual feedback on coalition formation, role specialization, reciprocity dynamics, and defection events — all without modifying your training code.

```bash
pip install alignscope
alignscope start
```

Then run any example:
```bash
python examples/test_kaz.py           # PettingZoo Knights-Archers-Zombies
python examples/test_mpe.py           # MPE (simple_spread, adversary, tag)
python examples/pettingzoo_example.py # Rock-Paper-Scissors
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

env = knights_archers_zombies_v10.env()
env = alignscope.wrap(env)  # ← one line, done

env.reset()
for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    action = env.action_space(agent).sample() if not (term or trunc) else None
    env.step(action)
```

**Tested with:** KAZ (Knights-Archers-Zombies), MPE (simple_spread, simple_adversary, simple_tag), Classic (Rock-Paper-Scissors, Connect Four), SISL, and Atari environments.

**What AlignScope auto-detects from PettingZoo:**
- **Roles** — inferred from agent names (`knight_0` → "knight", `adversary_0` → "adversary")
- **Teams** — inferred from naming conventions or adversarial keywords
- **Positions** — extracted from env internals (MPE physics, Pygame rects, or grid layout fallback)
- **Deaths** — detected by comparing alive agent sets between rounds


### 2. MPE (Multi-Particle Environments)

MPE environments like `simple_spread`, `simple_adversary`, and `simple_tag` are the benchmarks for cooperative and mixed-interest MARL. AlignScope extracts real continuous positions from the physics engine.

```python
from pettingzoo.mpe import simple_spread_v3
import alignscope

env = simple_spread_v3.env(N=4)
env = alignscope.wrap(env)
# Agents' real (x, y) positions appear on the topology graph
```

**What you see on the dashboard:**
- Agents moving in continuous 2D space
- Cooperative agents clustering near landmarks
- Adversaries (in `simple_tag`) chasing prey — visible as topology separation


### 3. SMAC (StarCraft Multi-Agent Challenge)

SMAC is the gold standard for cooperative MARL. AlignScope logs every battle step with real unit positions, health, and coordinated attacks.

```bash
# Requires StarCraft II installed
pip install pysc2 smac
python examples/test_smac.py
```

```python
# In test_smack.py — real SMAC integration:
from smac.env import StarCraft2Env
env = StarCraft2Env(map_name="2s3z")
# Logs real unit types (Stalker, Zealot), health, positions, and deaths
```

**What you see on the dashboard:**
- Unit role stability (Stalkers vs Zealots maintain distinct roles)
- Death events appear as red markers on the timeline
- Coordinated attacks show as topology edges between agents


### 4. RLlib (Ray)

RLlib is the most popular production RL framework. AlignScope plugs in as a standard RLlib callback.

```python
from ray.rllib.algorithms.ppo import PPOConfig
from alignscope.patches.rllib import AlignScopeCallback

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .callbacks(AlignScopeCallback)  # ← one line
)
algo = config.build()
algo.train()
```


### 5. EPyMARL (QMIX, MAPPO, VDN)

EPyMARL is the leading framework for value decomposition methods. AlignScope monkey-patches the EPyMARL logger to intercept training stats.

**Supported algorithms:** QMIX, MAPPO, VDN, COMA, IQL


### 6. Any Custom Environment

Don't use any of the above? AlignScope works with any Python code via the `alignscope.log()` API:

```python
import alignscope

alignscope.init(project="my-experiment")

for step in range(total_steps):
    obs, rewards, done = env.step(actions)

    alignscope.log(
        step,
        agents=[
            {"agent_id": 0, "team": 0, "role": "attacker", "x": 5.0, "y": 3.0,
             "energy": rewards[0], "is_defector": False, "coalition_id": 0},
            {"agent_id": 1, "team": 1, "role": "defender", "x": 8.0, "y": 7.0,
             "energy": rewards[1], "is_defector": False, "coalition_id": 1},
        ],
        actions=actions,
        rewards=rewards,
    )
```

## The Dashboard

AlignScope provides a real-time web dashboard at `http://localhost:8000` with three main panels:

### 1. Agent Topology (Left Panel)

A force-directed graph showing every agent as a node and their relationships as edges.

- **Nodes** represent individual agents, colored by team
- **Edges** appear when agents cooperate (e.g., `help_ally` actions)
- **Edge thickness** reflects the strength of the relationship (mutual help count)
- **Defecting agents** turn red and drift to the periphery
- **Click any node** to see detailed agent info (team, role, stability, coalition)

### 2. Cooperative Metrics (Right Sidebar)

Real-time alignment metrics computed from agent behavior:

| Metric | What It Measures | How It's Computed |
|--------|------------------|-------------------|
| **Overall Alignment Score** | How well all agents are cooperating | Weighted average of all metrics below |
| **Role Stability** | Are agents specializing? | Shannon entropy of each agent's role history — high stability = consistent roles |
| **Reciprocity Index** | Are agents helping each other back? | `min(help_given, help_received) / max(help_given, help_received)` for each pair |
| **Goal Convergence** | Are agents pursuing complementary actions? | Cosine similarity of action frequency vectors |
| **Coalitions** | How many cooperative groups exist? | Connected components in the help graph |
| **Defectors** | How many agents have broken cooperation? | Count of agents flagged as defectors |
| **Alignment History** | Visual trend of alignment over time | Sparkline chart of the overall alignment score |

### 3. Event Timeline (Bottom Panel)

A scrollable, zoomable timeline showing every significant event during training:

- **Defection events** (red) — the exact tick when an agent stops cooperating
- **Coalition fragmentation** (yellow) — when a team's coalition count drops
- **Reciprocity drops** (orange) — sudden breakdown in mutual help between agent pairs
- **Stability drops** — when an agent's role consistency suddenly changes

**Timeline features:**
- **Hover** over any event dot to see a detailed tooltip with the agent ID, team, and reason
- **Zoom** with `Shift + Scroll` or `Ctrl + Scroll` to inspect dense event regions
- **Scroll horizontally** to navigate through long training runs
- **Dynamic grid labels** that adapt spacing based on the total timeline length (100/500/1000 step intervals)

### 4. Header Bar

The top bar shows at-a-glance status:
- **Current Tick** — how far the simulation has progressed
- **Alignment Score** — the current overall alignment value
- **Event Count** — total number of detected anomalies
- **Connection Status** — whether the SDK is connected to the dashboard


## How Detection Works

AlignScope's `DefectionDetector` automatically identifies alignment-breaking events using **three independent methods**:

### 1. Direct Defection Events
When the environment reports an agent death or defection (e.g., a PettingZoo agent being terminated), AlignScope logs it immediately with a severity score and reason.

### 2. Metric-Based Anomaly Detection
The detector maintains a rolling window of metrics and flags sudden drops:
- **Reciprocity drop** — if mutual help between two agents drops by > 30% compared to the rolling average
- **Stability drop** — if an agent's role consistency drops by > 20%

### 3. Coalition Fragmentation
When the number of active coalitions within a team decreases (agents stop cooperating), the detector emits a fragmentation event.

All severities are computed **dynamically** — not hardcoded. For example, an agent death when only 2 agents remain is rated higher severity than when 8 agents are still alive.

## Integration Tiers

### Tier 1 — Zero Code (PettingZoo Auto-Wrap)
```python
env = alignscope.wrap(your_pettingzoo_env)
# done — every step auto-logged
```

### Tier 2 — One Line SDK
```python
alignscope.log(step, agents=agents, actions=actions, rewards=rewards)
```

### Tier 3 — Framework Callbacks
```python
# RLlib
config = PPOConfig().callbacks(AlignScopeCallback)

# EPyMARL — patches Logger.log_stat automatically
```

### Tier 4 — Offline Replay
```bash
python examples/test_log_replay.py --file my_training_run.csv
python examples/test_log_replay.py --file saved_run.npz
python examples/test_log_replay.py --wandb entity/project/run_id
```

Supports CSV, JSON, NPZ (numpy), Pickle, TensorBoard event files, and Weights & Biases runs.

## Project Structure

```
alignscope/
├── __init__.py          # Public API: init(), log(), wrap(), start()
├── sdk.py               # Core tracker — normalizes data, computes metrics, sends via WebSocket
├── server.py            # Dashboard server (FastAPI + WebSocket)
├── cli.py               # CLI: alignscope start / share
├── simulator.py         # Built-in demo data generator for --demo mode
├── metrics.py           # Alignment metrics engine (reciprocity, stability, convergence)
├── detector.py          # Anomaly detector (defections, reciprocity drops, fragmentation)
├── adapters.py          # Universal environment adapters (SMAC, MaMuJoCo, GRF, etc.)
├── patches/
│   ├── rllib.py         # RLlib callback integration
│   ├── pettingzoo.py    # Universal PettingZoo wrapper (any env)
│   └── pymarl.py        # PyMARL / EPyMARL logger patch
└── integrations/
    ├── wandb_bridge.py  # Auto-forward metrics to Weights & Biases
    └── mlflow_bridge.py # Auto-forward metrics to MLflow

frontend/
├── index.html           # Dashboard layout
├── css/style.css        # Dark-mode styling with custom scrollbars
└── js/
    ├── app.js           # Main controller — WebSocket ↔ UI binding
    ├── metrics.js       # Sidebar metrics rendering and sparklines
    ├── topology.js      # Force-directed agent graph (D3.js)
    └── timeline.js      # Zoomable event timeline (Canvas)

examples/
├── test_kaz.py              # Real KAZ environment (PettingZoo)
├── test_mpe.py              # Real MPE environments (simple_spread, adversary, tag)
├── test_smac.py             # Real SMAC environment (StarCraft II)
├── pettingzoo_example.py    # PettingZoo one-line wrapper demo (RPS)
├── test_rllib.py            # RLlib callback integration demo
└── test_log_replay.py       # Offline replay from CSV/JSON/NPZ/TensorBoard/W&B
```

## Quick Start

```bash
# 1. Install
pip install alignscope

# 2. Start the dashboard
alignscope start

# 3. Run a real MARL example (in a second terminal)
cd examples
python test_kaz.py

# 4. Open your browser
# http://localhost:8000
```

## Development

```bash
git clone https://github.com/raghavarajunithisha-lab/AlignScope.git
cd AlignScope
pip install -e ".[dev]"
alignscope start --demo
```

## Contact

If you encounter any bugs or issues, feel free to reach out:

📧 **nithisha2201@gmail.com**

If you'd like to **collaborate** and help develop AlignScope further — new environment adapters, better detection algorithms, or UI improvements — I'd love to hear from you. Contact me at the email above or open an issue on GitHub!

> **Note:** AlignScope is not yet published on PyPI. If there is enough interest from the community — whether it's collaborators, suggestions for improvements, or researchers who want to use it in their workflow — I plan to publish it as a proper PyPI package so anyone can install it with a simple `pip install alignscope`. Until then, you can install it locally with:
> ```bash
> git clone https://github.com/raghavarajunithisha-lab/AlignScope.git
> cd AlignScope
> pip install -e .
> ```



