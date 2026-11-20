# Roadmap

Build phases for Amoeba. Each phase adds the code and environments needed to run a group of experiments from the research agenda. Experiment IDs (E0.x, E1.x, …) refer to that agenda.

Environments live under `environments/src/amoeba_environments/`.

## Phase 0: Harness and baselines

Get the loop running end to end and measure its cost before measuring learning.

**Deliverables**
- PMDA loop in `agent.py`: stages, SQLite store, k-hop retrieval, relation operations
- Scorer backed by a llama.cpp server (prompt log-probabilities), with content-token masking
- `server.py`, `client.py`, `cli.py` implemented
- Per-step logging: percept, relation operations, surprise report, per-stage latency and tokens
- Genome loading from YAML with a `default` program

**Environments**
- `petri/`: a minimal grid world with energy, movement and eating, and one hidden rule. Exposes `H_true` for each observation.
- A null environment that replays a fixed observation stream, for repeatable benchmarks.

**Exit criteria**
- Latency per stage and per step measured for at least three model configurations (E0.1, E0.2)
- World-model growth and prompt size measured over long runs (E0.3)
- Scorer stability confirmed: identical inputs give identical bits (E0.4)

## Phase 1: Does the model stage help?

**Deliverables**
- Surprise gating (learn / revise / knowledge / ignore) and edit acceptance on a replay window
- Ablation switches: no Model stage (PDA), full-history memory, summary memory
- Evaluation scripts: excess-bits curves, per-relation credit, behavior scores

**Environments**
- Behavior-suite dishes: chemotaxis, avoidance, homeostasis, habituation, association
- Petri variants with several hidden rules of increasing difficulty

**Exit criteria**
- Excess bits fall within a lifetime in the single-rule dish (E1.1)
- PMDA beats PDA and the memory baselines at equal compute (E1.2, E1.3)
- ΔS stays positive on low-surprise steps (E1.4)

## Phase 2: Lifetime learning and new bodies

**Deliverables**
- Sleep phase: merge, decay, prune, and form relations about relations
- Relevance-weighted surprise with weights in the genome
- Curiosity term in Decide to avoid the dark-room problem

**Environments**
- Nonstationary dishes where rules change mid-life
- TextWorld adapter (comparison with AriGraph)
- ARC-AGI-3 adapter for the public games

**Exit criteria**
- Sleep keeps the model bounded without losing predictive value (E2.1)
- Cells recover after rule changes (E2.2)
- Measurable progress on at least one third-party environment (E2.3, E2.4)

## Phase 3: Populations and variation

**Deliverables**
- Fission with asymmetric partition of the relation store
- Genome mutation operators: numeric perturbation, LLM prompt rewrites, model-routing swaps, evolvable mutation rate
- Horizontal transfer and recombination
- Archive with quality-diversity selection (MAP-Elites style)

**Environments**
- Population dish: many cells, shared resources, energy-gated division and death

**Exit criteria**
- Weismannian, fission and Lamarckian inheritance compared (E3.1)
- Fitness improves across generations; diversity is maintained (E3.2, E3.3)

## Phase 4: Colonies

**Deliverables**
- Cell-to-cell messaging: local transport first, then Celluler over Pear/Holepunch
- Signal actions and morphogen-style broadcasts; shared colony energy
- Expression programs selected by signals; lateral inhibition
- Consensus on shared relations across cells

**Environments**
- Multi-resource dish: two resource types that need different senses
- Colony dish with adhesion and neighbor signaling

**Exit criteria**
- Division of labor emerges, and specialized colonies beat groups of generalists (E4.1, E4.2)
- Consensus models predict better than individual models (E4.3)

## Phase 5: Development and growth control

**Deliverables**
- Developmental program: one founder cell grows into a colony
- Target-composition homeostasis, gated division, apoptosis
- Defector (cheater) detection and resistance

**Environments**
- Development dish with damage events (removing a fraction of cells)

**Exit criteria**
- Colonies regrow their composition after losing 30% of cells (E5.1)
- Colonies resist defecting cells (E5.2)
- Whether a reproductive germ line emerges, or must be designed in (E5.3)

## Phase 6: Specialized substrates

**Deliverables**
- Per-cell or per-type LoRA adapters trained from long-confirmed relations during sleep
- Multi-adapter serving
- Multimodal models for sensory cell types
- Trained encoders and decoders for stages, distilled from logged traces

**Environments**
- Visual and audio dishes for sensory specialization

**Exit criteria**
- Consolidating relations into adapters preserves predictive value while shrinking context (E6.1)
- Adapter-specialized cell types outperform prompt-only specialization (E6.2)
