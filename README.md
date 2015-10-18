# Wishful Thinking: Simulations

Simulations used for exploring the effects of wishful thinking in decision making.

## Requirements

Running this requires Python 2.7, and is untested with other versions.
Install the requirements with `pip install -r requirements.txt`. It requires
NumPy, SciPy, and matplotlib.

## Instructions

See `python run.py -h` for now:

```
usage: run.py [-h] scenario model episodes

run simulations of agents with wishful thinking

positional arguments:
  scenario    the scenario to use (currently: sequence, fixed, gen_from_fixed,
              pmexp)
  model       the kind of model to use (currently: old, valopt)
  episodes    the number of episodes to run

optional arguments:
  -h, --help  show this help message and exit
```
