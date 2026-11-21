# amoeba-agent

Amoeba is a lightweight agent harness for small language models that run fast on local hardware. It asks two questions: how much intelligence can a well-designed harness get out of small models with limited resources, and what kind of intelligence emerges from a colony of such agents?

The goal is not a senior software engineer. The goal is an agent about as intelligent as an amoeba would be in a digital environment, and a foundation for building more complex agents out of many of them.

## The idea

- **A single cell.** Each Amoeba agent is one cell running a **perceive → model → decide → act** loop. It extends the perceive-decide-act loop from Donald Hoffman's interface theory of perception with an explicit modeling step.
- **The world model is the interface.** The cell knows nothing about the world directly. It builds a model of subject-predicate-object relations that lets it predict and act. The model is its memory and its identity.
- **Learning is driven by surprise.** A cell's model is doing its job if it makes the cell's next observation less surprising. Surprise decides when the cell learns new relations and when it consolidates what it already knows.
- **Embodiment.** A launched agent process is a cell's body. Environments push observations into it and receive actions back. The first environment is a petri dish with hidden rules to discover.
- **From cells to colonies.** Later, cells reproduce with variation, specialize, and combine into colonies. That work connects to the Celluler protocol.

## Status

Pre-alpha. The basic PMDA loop and relational world model run end to end. Surprise scoring is not implemented yet.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-packages
uv run amoeba --help
uv run pytest
```

### Running a cell against a local model

The cell talks to any OpenAI-compatible chat server with JSON-schema output, such as
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. The default genome
expects one at `http://127.0.0.1:8080/v1`.

```bash
# 1. Serve a small model (any GGUF works; smaller is faster)
llama-server -m LFM2.5-1.2B-Instruct-Q4_K_M.gguf --port 8080

# 2a. Run a cell in-process on the 1-D line environment, printing per-stage timings
uv run python -m amoeba_environments.line --steps 20

# 2b. Or launch a cell as an API process and talk to it
uv run amoeba run --db alpha0.db --port 8700
uv run amoeba step "You smell food to your right. Energy: 10." -a move_left -a move_right -a eat
uv run amoeba model food
uv run amoeba state
```

To use another server or model per stage, copy `src/amoeba/genome.default.yaml`, edit its
`base_url` or `models` section, and pass it with `--genome`.

## Layout

```
src/amoeba/           core library
  agent.py            the cell: PMDA loop, world model, genome (single file)
  genome.default.yaml default genome: stage prompts, model routing, params
  server.py           API process hosting one cell
  client.py           client for a running cell
  cli.py              the `amoeba` command
environments/         workspace package: environments that embody cells
  src/amoeba_environments/line.py    1-D line with food: loop smoke test and timings
  src/amoeba_environments/petri/     petri dish (placeholder)
experiments/          standalone experiments from the research agenda
  translation/        E0.0: can local models perceive into the internal language?
tests/
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
