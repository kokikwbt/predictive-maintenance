# Dataset notebooks

This directory is reserved for rebuilt dataset-specific notebooks. Dataset
files must be loaded from `../../data/raw/<dataset-id>` and must not be stored
beside metadata or library code under `datasets/`.

Keep these notebooks focused on:

- loading and validating the source data;
- explaining columns, entities, labels, and time axes;
- visualizing representative samples and class distributions;
- demonstrating dataset-specific helper functions.

Task-level modeling and evaluation examples belong in `../tasks/`.
