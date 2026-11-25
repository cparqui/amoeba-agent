# E0.0: Internal-language translation

Can small off-the-shelf models, without fine-tuning, translate raw observations into the cell's internal language? This is the first experiment in the research agenda. It runs without the agent harness: only a script, a set of observations and some local models.

The internal language is made of relations, usually `(predicate subject object)`. The only built-ins are `and`, `or` and `not`. Time, confidence and source are ordinary predicates that wrap a statement, e.g. `(at_time 7 (touches ?x1 self))`. Unidentified things become variables (`?x1`). Predicates are not fixed in advance: the perceiver reuses symbols from its lexicon or coins new ones.

## Conditions

| Factor | Values |
|---|---|
| Model | any OpenAI-compatible endpoint (local llama.cpp servers, vLLM, hosted) |
| Syntax | `sexpr` `(on cup table)`, `prolog` `on(cup, table).`, `json` `["on", "cup", "table"]` |
| Lexicon | `empty`; `seeded` (fixed seed list); `evolving` (own earlier symbols fed back); `seeded_evolving` |
| Decoding | `free`; `constrained` (GBNF grammar per syntax; llama.cpp and vLLM only) |

Each condition runs over `data/observations.jsonl` in order, `repeats` times. The 32 observations mix event logs, text and JSON. They include six paraphrase groups (the same fact said 3–4 ways, in different formats) spread through the sequence, three observations with unidentifiable entities, and a short question with accepted answers for each observation.

## Metrics

| Metric | Meaning |
|---|---|
| `parse_rate` | outputs that parse in the condition's syntax |
| `consistency` | mean pairwise overlap (Jaccard) of canonical facts within each paraphrase group |
| `pred_consistency` | the same over predicate names only (looser) |
| `stability` | overlap of the same observation's facts across repeats |
| `reuse_rate` | output symbols already in the lexicon before that step |
| `lexicon_final`, `convergence` | final lexicon size; new symbols in the last third ÷ the first third (below 1 = converging) |
| `novel_var_rate` | unidentified entities given a variable rather than a name |
| `fidelity` | questions a judge answers correctly from the translation alone; compare with the `(raw input)` row |
| latency, tokens | p50/p95 seconds and mean completion tokens per observation |

Facts are canonical: modifier nesting order is ignored, symbols are case- and separator-insensitive, and variables are anonymized. `(at_time 1 (conf 0.8 (on cup table)))` and `(conf 0.8 (at_time 1 (on cup table)))` are the same fact. See `syntax.facts`.

Target (from the agenda): one local model with parse rate ≥ 95%, consistency ≥ 0.8 with an evolving lexicon, convergence below 1, and fidelity within 10 points of the raw input.

## Running it

1. Install [llama.cpp](https://github.com/ggml-org/llama.cpp) so `llama-server` is on your `PATH` (or set `LLAMA_SERVER=/path/to/llama-server`). You don't start any servers yourself: each endpoint with a `serve:` block is started on a free port, health-checked, used and stopped by the pipeline. `-hf` models download from Hugging Face on first start.

2. Configure, and optionally check that every model starts and answers:

   ```bash
   cp experiments/translation/config.example.yaml experiments/translation/config.yaml
   uv run python -m experiments.translation check -c experiments/translation/config.yaml
   ```

3. Run everything with one command:

   ```bash
   uv run python -m experiments.translation pipeline -c experiments/translation/config.yaml
   ```

   For each model in turn it starts the server, translates under every condition, and stops the server. Then it starts the judge, scores fidelity, writes the summary and, if tracking is configured, logs it all to MLflow. Models run one at a time so that latency isn't skewed by other models sharing the hardware; `--parallel` runs them together. Other options: `--no-judge`, `--no-track`, `--resume runs/e0.0/<stamp>`.

   With `--resume`, models whose conditions are all finished are skipped without starting a server. Partly finished condition repeats are rerun from the start, because an evolving lexicon depends on every earlier step.

   The stages also run on their own (each starts the servers it needs):

   ```bash
   uv run python -m experiments.translation run   -c experiments/translation/config.yaml
   uv run python -m experiments.translation judge runs/e0.0/<stamp> -c experiments/translation/config.yaml
   uv run python -m experiments.translation score runs/e0.0/<stamp>
   ```

The defaults (4 models less the judge, 3 syntaxes, 3 lexicon modes, 2 decodings, 3 repeats, 32 observations) make about 1,700 calls per model. Narrow `syntaxes`, `lexicon_modes` or `repeats` for a first pass.

### Tracking

Set `tracking.uri` in the config (or `MLFLOW_TRACKING_URI`) to an MLflow server, e.g. `http://workstation:5000`. Each pipeline run becomes one MLflow run with the condition grid as params and git commit, host and run dir as tags. It has one nested run per condition (`model/syntax/lexicon/decoding`, plus `raw-input`) holding that row's metrics, so conditions can be compared side by side in the MLflow UI. The whole run dir is attached as artifacts. The client is `mlflow-skinny`, installed by `uv sync`. An MLflow server started with `--artifacts-destination <local dir>` stores artifacts on its own disk, so no S3 is needed.

Outputs in the run dir: `results.jsonl` (one record per call: raw output, parsed form, facts, symbols, latency), `judge.jsonl`, `summary.csv`, `summary.md`, `provenance.json` (git commit and dirty flag, host, platform, Python, package versions), `servers.json` (per model: exact command, llama-server version, startup time, `/props`) and `servers/<model>.log`. The results file is also the dataset for fine-tuning a perceiver later.

## Notes

- **Inference is backend-agnostic.** Everything speaks the OpenAI-compatible chat API. llama.cpp is the default local backend because it runs GGUF models on CPU, Apple silicon and GPUs, and takes a GBNF `grammar` per request. vLLM works with `backend: vllm` (use `grammar_field: guided_grammar` on older versions). Ollama, LM Studio and hosted APIs work with `backend: openai`, free decoding only.
- **Thinking models.** Turn reasoning off per endpoint with `extra: {chat_template_kwargs: {enable_thinking: false}}`. Any `<think>` blocks left in the output are stripped before parsing.
- **Constrained decoding can hurt.** Grammars guarantee parseable output but may lower quality (see "Let Me Speak Freely?", Tam et al. 2024). That trade-off is why decoding is a factor here.
