# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a deep reinforcement learning fundamentals practice repository focused on exact methods for tabular MDPs. The project implements Value Iteration, Policy Iteration, and Soft (Max-Entropy) Value Iteration algorithms with comprehensive testing and visualization capabilities.

## Common Commands

### Building and Testing
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"    # Skip slow tests
pytest -m "grid"        # Run only gridworld tests

# Run tests with coverage
pytest --cov=src

# Run single test file
pytest tests/test_value_iteration.py
pytest tests/test_policy_iteration.py::test_pi_toy2state_closed_form
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Generating Figures
```bash
# Individual algorithm figures
make fig-vi           # Value Iteration
make fig-pi           # Policy Iteration  
make fig-soft         # Soft Value Iteration
make fig-all          # All three algorithms

# With slip probability variations
make fig-vi-slip03
make fig-pi-slip03

# Reproduce all results
./reproduce.sh
```

### Running Algorithms
```bash
# Set up environment
export PYTHONPATH=src

# Run training with Hydra configuration
cd exact-methods/
python -m rlx.scripts.train           # Uses default config
python -m rlx.scripts.train algo=dp/policy_iteration
python -m rlx.scripts.train env=tabular/gridworld algo.tol=1e-10

# Generate specific visualizations
python -m rlx.scripts.make_4room_figure --algo vi --tol 1e-8 --outfile custom.png
python -m rlx.scripts.make_4room_figure --algo soft_vi --tau 0.1 --slip 0.3
```

## Code Architecture

### Core Structure
- **`rlx/`**: Main package containing algorithms and environments
- **`rlx/envs/tabular/`**: Tabular MDP environments and utilities
- **`rlx/algos/dp/`**: Dynamic programming algorithms (VI, PI, Soft-VI)
- **`rlx/scripts/`**: Training scripts and visualization tools
- **`tests/`**: Comprehensive test suite with correctness and performance tests
- **`conf/`**: Hydra configuration files for experiment management

### Key Components

#### TabularMDP Class (`rlx/envs/tabular/mdp.py`)
Central abstraction for all tabular MDPs with:
- `P`: Transition probabilities (S, A, S)
- `R`: Immediate rewards (S, A) 
- `gamma`: Discount factor
- `terminal_mask`: Boolean mask for terminal states
- Optional metadata for visualization and debugging

#### Algorithm Implementations
All algorithms follow consistent interface patterns:
- **Input**: TabularMDP, tolerance, max_iterations, optional logger
- **Output**: Dictionary with `V` (values), `Q` (action-values), `pi` (policy), `logs` (metrics), `run_time`
- **Logging**: Standardized per-iteration metrics including Bellman residuals, policy changes, entropy, wall-clock time

#### Environment Builders
- **Toy2State**: Simple 2-state MDP with closed-form solutions for testing
- **4-Room Gridworld**: Standard benchmark with configurable slip probability and step penalties
- **ASCII-based builders**: Flexible gridworld construction from text layouts
- **Cliff Walking**: Classic RL environment with pit states

### Testing Strategy

The test suite implements the acceptance criteria from CLAUDE.md:
- **Correctness**: VI matches closed-form solutions on Toy2State within 1e-8
- **Agreement**: VI and PI produce identical results within 1e-6
- **Convergence**: All algorithms converge within specified tolerances and iteration limits
- **Monotonicity**: Bellman operators exhibit expected mathematical properties
- **Soft Limits**: Soft-VI approaches hard VI as temperature → 0
- **Reproducibility**: Identical seeds produce identical results

### Configuration System

Uses Hydra for hierarchical configuration management:
- **`conf/config.yaml`**: Top-level defaults
- **Algorithm configs**: Individual parameter sets for VI, PI, Soft-VI
- **Environment configs**: MDP construction parameters
- **Overrides**: Command-line parameter specification

### Key Implementation Details

#### Numerical Stability
- Uses `logsumexp` for soft value iteration to prevent overflow
- Implements both naive and gamma-scaled convergence criteria
- Handles tie-breaking in argmax operations consistently

#### Performance Monitoring
- Tracks Bellman residuals, policy changes, and entropy across iterations
- Wall-clock timing for algorithm comparison
- CSV logging for experiment reproducibility

#### Visualization
- Policy arrow plotting for gridworlds
- Value function heatmaps with customizable colormaps
- Convergence curve generation with standardized metrics

## Development Notes

- **PYTHONPATH**: Always set `PYTHONPATH=src` when running modules
- **Working Directory**: Run commands from `exact-methods/` subdirectory
- **Test Organization**: Use pytest markers (`@pytest.mark.slow`, `@pytest.mark.grid`) for selective test execution
- **Reproducibility**: Use `reproduce.sh` to generate all figures with consistent parameters
- **Configuration**: Prefer Hydra configs over hardcoded parameters for experiments

## Important Files

- **`CLAUDE.md`**: Contains the complete L1 assignment specification with acceptance criteria
- **`Makefile`**: Convenient targets for figure generation
- **`reproduce.sh`**: Script to regenerate all experimental results
- **`pytest.ini`**: Test configuration with custom markers and paths
