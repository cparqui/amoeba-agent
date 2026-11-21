"""Line: the simplest embodiment, a cell on a 1-D line with one food source.

No hidden rules and no H_true; this environment exists to exercise the PMDA
loop end to end and to print per-step timings (Phase 0 baselines).

    uv run python -m amoeba_environments.line --steps 20
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
from dataclasses import dataclass, field

from amoeba import ActionSpec, Agent, Genome, Observation, StepResult, Store

AFFORDANCES = [
    ActionSpec(name="move_left", description="move one cell left (costs 1 energy)"),
    ActionSpec(name="move_right", description="move one cell right (costs 1 energy)"),
    ActionSpec(name="eat", description="eat food at your position (gains 5 energy if food is here)"),
]


@dataclass
class LineWorld:
    """A line of `size` cells; food sits at one position and respawns when eaten."""

    size: int = 9
    energy: int = 10
    seed: int = 0
    position: int = 0
    food: int = 0
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.position = self.size // 2
        self.food = self._spawn()

    def _spawn(self) -> int:
        return self.rng.choice([i for i in range(self.size) if i != self.position])

    @property
    def alive(self) -> bool:
        return self.energy > 0

    def observe(self, last_event: str = "") -> Observation:
        if self.food == self.position:
            smell = "There is food right here."
        else:
            side = "right" if self.food > self.position else "left"
            smell = f"You smell food to your {side}, {abs(self.food - self.position)} cells away."
        text = f"{last_event} You are at position {self.position} of {self.size - 1}. {smell} Energy: {self.energy}."
        return Observation(text=text.strip(), affordances=AFFORDANCES)

    def apply(self, action: str) -> str:
        """Advance the world by one action; returns a short event description."""
        if action in ("move_left", "move_right"):
            self.energy -= 1
            target = self.position + (1 if action == "move_right" else -1)
            if 0 <= target < self.size:
                self.position = target
                return f"You moved {action.removeprefix('move_')}."
            return "You bumped into the edge."
        if action == "eat":
            if self.food == self.position:
                self.energy += 5
                self.food = self._spawn()
                return "You ate the food. It was good."
            self.energy -= 1
            return "There was nothing to eat."
        self.energy -= 1
        return "Nothing happened."


async def run(agent: Agent, world: LineWorld, steps: int, verbose: bool = True) -> list[StepResult]:
    """Embody `agent` in `world` for up to `steps` steps (stops if energy runs out)."""
    results: list[StepResult] = []
    event = ""
    for _ in range(steps):
        if not world.alive:
            break
        result = await agent.step(world.observe(event))
        event = world.apply(result.action.name)
        results.append(result)
        if verbose:
            t = result.timings_ms
            tokens = sum(u.prompt_tokens + u.completion_tokens for u in result.usage.values())
            print(
                f"step {result.step:>3}  {result.action.name:<10} energy={world.energy:<3} "
                f"P={t['perceive']:.0f} M={t['model']:.0f} D={t['decide']:.0f} A={t['act']:.0f} "
                f"total={t['total']:.0f}ms  tokens={tokens}  ops={result.ops_applied}  "
                f"relations={result.context.relations_total}  db={result.context.db_bytes}B"
            )
    if verbose and results:
        print(f"mean step: {statistics.mean(r.timings_ms['total'] for r in results):.0f}ms over {len(results)} steps")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Amoeba cell on a 1-D line with food.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--genome", default=None, help="genome YAML (default: packaged)")
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--base-url", default=None, help="override the genome's model server URL")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    genome = Genome.load(args.genome)
    if args.base_url:
        genome.base_url = args.base_url
    agent = Agent(genome, Store(args.db), cell_id="line")
    asyncio.run(run(agent, LineWorld(seed=args.seed), args.steps))


if __name__ == "__main__":
    main()
