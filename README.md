# vae-cancer-nodules

## Development

1. Clone repo
2. Install `conda` with `https://docs.conda.io/projects/conda/en/latest/user-guide/install/`
3. Install `conda` environment from `conda_requirements.yaml` with
`$conda env create -f requirements-env.yaml`. Environment name can be changed
in the `name:` field of yaml. By default it is `vae-cancer-nodules`
4. Activate the environment: `conda activate vae-cancer-nodules`
5. Install the git pre-hook with `pre-commit install` from the root directory.
6. Enjoy the science!

For documenting the code you can use the great extension for VSCode: autoDocstring
