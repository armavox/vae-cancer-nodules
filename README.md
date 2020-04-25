# rls-med

## Development

1. Clone repo
2. Install `conda` with `https://docs.conda.io/projects/conda/en/latest/user-guide/install/`
3. Install `conda` environment from `conda_requirements.yaml` with
`$conda env create -f requirements-env.yaml`. Environment name can be changed
in the `name:` field of yaml. By default it is `rls-med`
4. Activate the environment: `conda activate rls-med`
5. Install the git pre-hook with `pre-commit install` from the root directory.
6. Enjoy the science!

> **IMPORTANT NOTE**
> 
> There bug in pylidc with DICOM pixel_aray preprocessing. Until pylidc:master will be updated use my branch:
> ```bash
> pip install pylidc
> git clone https://github.com/armavox/pylidc.git && cd pylidc
> git checkout fix_slope_intercept && cd ..
> pip install -U ./pylidc

For documenting the code you can use the great extension for VSCode: autoDocstring
