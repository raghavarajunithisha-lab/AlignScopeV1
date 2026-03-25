# Why AlignScope Matters for Alignment Research

AI alignment is not just a single-agent problem. As multi-agent reinforcement learning (MARL) systems grow in complexity — agents learning to cooperate, specialize, and form coalitions — we need tools to **observe** alignment as it emerges.

AlignScope is a scientific instrument for this phenomenon.

## The Measurement Gap

You can't study what you can't see.

Most MARL training pipelines produce experience logs and reward curves that tell you *whether* an agent won, but not *how* alignment formed — or when it broke. The three questions that matter most to alignment research are currently invisible in standard tooling:

1. **Are agents specializing?** When agents commit to consistent roles, that's not just strategy — that's the beginning of role-based trust, the same dynamic that makes cells differentiate in an organism. AlignScope measures this as **role stability** via Shannon entropy.

2. **Are agents reciprocating?** When Agent A gathers resources for Agent B, and B captures objectives for A, that's the seed of coalition — individuals learning to form cooperative wholes. AlignScope measures this as the **reciprocity index**.

3. **When does alignment fail?** The most scientifically valuable moment in any training run is not when everything works — it's the exact tick when an agent defects. That moment reveals what the alignment was actually built on, and how fragile it was. AlignScope's **defection detector** flags these moments so researchers can inspect what changed.

## What This Enables

With AlignScope, a researcher can:

- Watch coalition clusters form in real-time in the topology graph and see which role combinations naturally attract
- Identify which training configurations produce stable alignment vs. fragile cooperation that breaks under pressure
- Compare episodes side-by-side using alignment metric histories rather than just reward curves
- Build intuition about organic alignment by *seeing* it happen, not reconstructing it from logs

## Environment Agnostic

AlignScope works with **any** MARL environment. Whether you're training agents in grid worlds, competitive games, robotics simulations, or social dilemma environments — if your agents have roles, teams, and interactions, AlignScope can visualize their alignment dynamics.

This is not observability for observability's sake. It's a scientific instrument for studying emergent cooperation in artificial intelligence.
