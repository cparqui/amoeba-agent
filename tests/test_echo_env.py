import asyncio

from amoeba_environments.echo import LineWorld, run

from amoeba.agent import Agent


def test_echo_runs_in_process(agent: Agent) -> None:
    world = LineWorld(seed=1)
    start = world.position
    results = asyncio.run(run(agent, world, steps=3, verbose=False))
    assert len(results) == 3
    assert all(r.action.name == "move_right" for r in results)
    assert world.position == min(start + 3, world.size - 1)


def test_line_world_rules() -> None:
    world = LineWorld(seed=0, size=5, energy=3)
    world.food = world.position
    assert "food right here" in world.observe().text
    assert world.apply("eat").startswith("You ate") and world.energy == 8
    assert world.food != world.position
