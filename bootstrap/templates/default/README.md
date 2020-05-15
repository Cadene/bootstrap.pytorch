# {PROJECT_NAME}

## Install

[Conda](https://docs.conda.io/en/latest/miniconda.html)

```bash
conda create --name {PROJECT_NAME_LOWER} python=3
source activate {PROJECT_NAME_LOWER}

cd $HOME
git clone --recursive https://github.com/{PROJECT_NAME}/{PROJECT_NAME_LOWER}.bootstrap.pytorch.git
cd {PROJECT_NAME_LOWER}.bootstrap.pytorch
pip install -r requirements.txt
```

## Reproducing results

Run experiment:
```bash
python -m bootstrap.run \
-o {PROJECT_NAME_LOWER}/options/{PROJECT_NAME_LOWER}.yaml \
--exp.dir logs/{PROJECT_NAME_LOWER}/1_exp
```

Display training and evaluation figures:
```bash
open logs/{PROJECT_NAME_LOWER}/1_exp/view.html
```

Display table of results:
```bash
python -m bootstrap.compare -o 