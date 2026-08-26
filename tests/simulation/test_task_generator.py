import inspect
from collections import Counter

from embedding_space_exploration.simulation.generator import simulate_embeddings
from embedding_space_exploration.simulation.task_generator import (
    BASE,
    FRAMES,
    N_REPLICATES,
    calibration_grid,
)


def test_every_condition_is_drawn_at_every_seed():
    grid = calibration_grid()
    conditions = Counter(cell_id.rsplit("-s", 1)[0] for cell_id in grid)
    assert set(conditions.values()) == {N_REPLICATES}
    assert len(grid) == len(conditions) * N_REPLICATES

    seeds = {cell_id: kwargs["random_state"] for cell_id, kwargs in grid.items()}
    assert len(set(seeds.values())) == N_REPLICATES


def test_every_cell_is_a_valid_call_to_the_generator():
    # The grid is data, so a mistyped knob would otherwise only surface as a
    # TypeError partway through a 129-cell build.
    accepted = set(inspect.signature(simulate_embeddings).parameters)
    for kwargs in calibration_grid().values():
        assert set(kwargs) <= accepted


def test_each_sweep_moves_exactly_one_knob():
    # The property that makes a sweep readable as one curve: its conditions may
    # hold several knobs off BASE (the coupling sweep pins a confound loading
    # C1 can see), but only one may *vary* along the sweep.
    sweeps = {}
    for cell_id, kwargs in calibration_grid().items():
        condition = cell_id.rsplit("-s", 1)[0]
        sweeps.setdefault(condition.rsplit("-", 1)[0], {})[condition] = kwargs

    for sweep, conditions in sweeps.items():
        if len(conditions) < 2:
            continue
        keys = set().union(*(kwargs.keys() for kwargs in conditions.values()))
        varying = {
            key
            for key in keys - {"random_state"}
            if len({kwargs.get(key) for kwargs in conditions.values()}) > 1
        }
        assert len(varying) == 1, f"{sweep} varies {varying}"


def test_the_structurally_distinct_cells_build():
    grid = calibration_grid()
    for cell_id in ("continuum-6.0-s0", "confound-radial-8.0-s0", "imbalance-rare-s0"):
        space = simulate_embeddings(**grid[cell_id])
        assert set(FRAMES) <= set(space)
        assert len(space["embeddings"]) == BASE["n_patients"]
