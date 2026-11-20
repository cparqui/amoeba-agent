# Roadmap

Build phases for Amoeba. Each phase adds the code and environments needed to run a group of experiments from the research agenda. Experiment IDs (E0.x, E1.x, …) refer to that agenda.

Environments live under `environments/src/amoeba_environments/`.

## Phase 0: Harness and baselines

Get the loop running end to end and measure its cost before measuring learning.

**Deliverables**
- `experiments/translation/`: the standalone E0.0 run (no harness) that tests whether local models can perceive into the internal s-expression language, and sets up the first local llama.cpp models
- PMDA loop in `agent.py`: stages, SQLite store, k-hop retrieval, relation operations
- Scorer backed by a llama.cpp server (prompt log-probabilities), with content-token masking
- `server.py`, `client.py`, `cli.py` implemented
- Per-step logging: percept, relation operations, surprise report, per-stage latency and tokens
- Genome loading from YAML with a `default` program

- Benchmark harness: concurrent cells per machine, resident memory and disk size per cell

**Environments**
- `petri/`: a minimal grid world with energy, movement and eating, and one hidden rule. Exposes `H_true` for each observation.
- A null environment that replays a fixed observation stream, for repeatable benchmarks.
- Calibration streams: fully deterministic (`H_true = 0`) and fully random (known `H_true`).

**Exit criteria**
- E0.0 Internal-language translation: a local model with parse rate ≥ 95%, paraphrase consistency ≥ 0.8 and a converging lexicon, or a decision to put a fine-tuned perceiver on the critical path
- E0.1 Loop latency per stage: stable latency distributions; dominant stage identified
- E0.2 Throughput and per-stage routing: a default routing configuration recorded
- E0.3 Scoring cost versus generation cost: scoring under 20% of step time
- E0.4 Prompt ordering and cache reuse: an ordering and S_base sampling rate where ΔS costs under 25% extra
- E0.5 Scorer noise floor and calibration: scorer chosen; noise well under 0.1 bits per token
- E0.6 World-model growth: sublinear growth with decay; context tokens plateau within budget
- E0.7 Memory footprint per cell: cost curve recorded for sizing Phase 3–5 populations

## Phase 1: Single-cell learning

**Deliverables**
- Surprise gating (learn / revise / knowledge / ignore) and edit acceptance on a replay window, with a held-out check and size penalty λ
- Per-relation credit from sampled leave-one-out scoring
- Relevance-weighted surprise with weights in the genome
- Ablation switches: no Model stage (PDA), full-history memory, summary memory
- Evaluation scripts: excess-bits curves, credit reports, graph-overlap comparison between cells

**Environments**
- Petri variants with 1, 3 and 5 hidden rules, conditional rules, and irreducibly random events
- Rule-change dish: rule A for the first half of life, rule B for the second, with A-probes
- Distractor dish: mostly energy-irrelevant observation content plus a few relevant cues

**Exit criteria**
- E1.1 Surprise curve with one hidden rule: S falls over the lifetime and the rule appears in the model
- E1.2 Excess bits toward zero: excess bits fall toward zero without going negative
- E1.3 Model value on low-surprise steps: ΔS positive on rule-governed low-surprise steps
- E1.4 PMDA versus alternatives at equal compute: PMDA beats PDA and summary memory (headline result)
- E1.5 Size penalty sweep: an intermediate λ minimizes held-out excess bits
- E1.6 Forgetting under rule change: the held-out check limits forgetting
- E1.7 Relevance-weighted versus uniform surprise: equal or higher fitness with a smaller model
- E1.8 Ontology divergence: similar excess bits across cells with low graph overlap

## Phase 2: Lifetime learning and embodiment

**Deliverables**
- Curiosity term in Decide to avoid the dark-room problem
- Sleep phase: merge, decay, prune, and form relations about relations
- Third-party environment adapters and baselines (AriGraph-style memory, graph-exploration baseline)

**Environments**
- Behavior-suite dishes: chemotaxis, avoidance, homeostasis, habituation, association, each with scripted-optimal and random reference policies
- Dark-room dish: a safe, unchanging corner and a riskier, food-rich open area
- Long-lifetime dishes with 3 or more rules for 50,000-step runs
- TextWorld adapter (comparison with AriGraph)
- ARC-AGI-3 adapter for the public games

**Exit criteria**
- E2.1 Amoeba behavior suite: criterion reached on all five; PMDA beats PDA on habituation and association
- E2.2 Dark room and curiosity: a positive curiosity weight maximizes fitness
- E2.3 Sleep consolidation: size stays bounded with no rise in excess bits
- E2.4 Second embodiment, text games versus AriGraph: PMDA matches or beats it at small model sizes
- E2.5 Third embodiment, ARC-AGI-3: more levels or fewer actions than the exploration-only baseline

## Phase 3: Populations and variation

**Deliverables**
- Fission with asymmetric partition of the relation store
- Genome mutation operators: numeric perturbation, LLM prompt rewrites, model-routing swaps, evolvable mutation rate
- Horizontal transfer and recombination
- Archive with quality-diversity selection (MAP-Elites style)

**Environments**
- Population dish: many cells, shared resources, energy-gated division and death; stable and periodically changing variants; more than one way to survive
- Neighbor messaging within the population dish, for horizontal transfer

**Exit criteria**
- E3.1 Inheritance schemes: Weismannian, fission and Lamarckian compared; default chosen
- E3.2 Mutation operators and evolvable mutation rate: at least one operator gives sustained gains
- E3.3 Quality-diversity versus fitness-only selection: the archive preserves niches and recovers faster
- E3.4 Horizontal transfer: useful rules spread faster without harmful ones spreading unchecked

## Phase 4: Colonies

**Deliverables**
- Cell-to-cell messaging: local transport first, then Celluler over Pear/Holepunch
- Signal actions and morphogen-style broadcasts; shared colony energy
- Expression programs selected by signals; lateral inhibition
- Consensus on shared relations across cells

**Environments**
- Multi-resource dish: two resource types that need different senses
- Colony dish with adhesion and neighbor signaling

- Partial-view dish: cells observe different parts of the same environment

**Exit criteria**
- E4.1 Division of labor: differentiating colonies develop stable types and beat generalists
- E4.2 Lateral inhibition: type ratios move closer to the optimum
- E4.3 Consensus as a shared interface: surprise-minimizing consensus beats naive merging and no sharing

## Phase 5: Development and growth control

**Deliverables**
- Developmental program: one founder cell grows into a colony
- Target-composition homeostasis, gated division, apoptosis
- Defector (cheater) detection and resistance

**Environments**
- Development dish with damage events (random and type-targeted removal of 30% of cells)
- Defector insertion into healthy colonies
- Multi-colony competition with colony-level selection

**Exit criteria**
- E5.1 Development from one founder cell: colonies reach and hold a target composition
- E5.2 Regeneration: with a setpoint, colonies recover after losing 30% of cells
- E5.3 Cheater resistance: at least one control mechanism contains defectors
- E5.4 Germ-line emergence: whether a reproductive type emerges, or must be designed in

## Phase 6: Substrates

Weight consolidation (E6.1, E6.2) can start as soon as E2.3 sleep consolidation works, in parallel with Phases 3–5.

**Deliverables**
- Per-cell LoRA adapters trained during sleep, from long-confirmed relations and from high-surprise transitions
- Per-type adapters and multi-adapter serving
- Multimodal models for sensory cell types
- Trained encoders and decoders for stages, distilled from logged traces

**Environments**
- Multisensory dish: one resource visible only in images, another only in sound

**Exit criteria**
- E6.1 Graph-to-weights consolidation: ΔS of consolidated relations falls to about zero; removing them leaves excess bits unchanged
- E6.2 Adapters learned from surprising transitions: beats graph-only learning without more forgetting than E1.6
- E6.3 Per-type adapters for specialized cells: improve fitness at modest cost
- E6.4 Multimodal sensory cells: fitness gain justifies the latency cost
- E6.5 Trained encoders and decoders per stage: match general-model learning at a fraction of the latency
