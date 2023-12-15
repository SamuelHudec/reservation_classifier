# Car recommender using public dataset

## Environment setup

To use this library/repo, make sure that you have Python >= `3.9.*` - we recommend using [pyenv][] for managing different
Python versions (see: [how to install pyenv on MacOS][]).

This project defines dependencies in [pyproject.toml](./pyproject.toml) according to [PEP-621](https://peps.python.org/pep-0621/)
For development create virtual env with:
```bash
python -m venv venv
source venv/bin/activate
```
Then run:
```bash
make install
```

# EDA

I tried to leave all my reflections and thoughts in the comments in the [notebook](./notebooks/solution_sketch.ipynb). Of course, there was no time to go
really deep and I tried to leave only the essentials.

# Architecture

I split the data preprocessing and training, even though I could use the sklearn pipeline and have the
whole process under one. On the other hand, this way we can easily experiment with models. For example, we can
replace xgb class with torch NN class or anything else. The same applies to encoding.

## Train

Now, I did not use grid search, but the best set of parameters. XGB is a sufficiently robust solution
and there is no need to search for new parameters in every training session.

```bash
python -m scripts.train --path_to_data=data/train/hotel_bookings.csv --output_model_path=data/test_model
```

Each run trains the model and at the end checks if the model is better based on the specified threshold.
If so, the model will be copied to the "best" folder. In this way, we will keep each trained model in case of
need, for example, inspections, etc.

## Predict in chunks

I left the output in numpy format because I don't know what would consume the output. At the same time, from the
beginning, the data lacks primary keys or at least the ID of the record. For prediction in chunks, I would suggest
using something different (dask, etc.), but pandas is sufficient for the task.

```bash
python -m scripts.predict --path_to_data=data/train/hotel_bookings.csv --output_path=data/predicted --model_path=data/test_model
```

# To Do
* write abstract classes for encoding and training,
* write dataclasses for handling hyperparameters,
* unit tests,
* marge GetParseData and GetPredictData,


[pyenv]: https://github.com/pyenv/pyenv#installationbrew
[how to install pyenv on MacOS]: https://jordanthomasg.medium.com/python-development-on-macos-with-pyenv-2509c694a808
[generative AI]: https://arxiv.org/abs/2305.05065