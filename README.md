<div align="center">

# Mixed-effects Transformers

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2205.01749)

</div>

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/juliaiwhite/mixed-effects-transformers.git
cd mixed-effects-transformers

# [OPTIONAL] create conda environment
conda myenv create -f environment.yml

# install pytorch according to instructions
# https://pytorch.org/get-started/
```

Install supported datasets (Reddit, Amazon Reviews, Movie Dialogue, or C4)

```bash
cd dataset_name
python generate_data.py
```


Train an MET model on the reddit dataset with the default configuration

```bash
# train on CPU
python src/train.py trainer=cpu
python run.py experiment=hierarchical/reddit.yaml trainer=cpu

# train on GPU
python run.py experiment=hierarchical/reddit.yaml trainer=gpu
```

Train any model (prefixtuned, finetuned, conditional, or hierarchical/MET) on one of the four supported datasets (Reddit, Amazon Reviews, Movie Dialogue, or C4) with an experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=model_name/dataset_name.yaml
```

You can override any parameter from command line (this command will train a finetuned model on thriller movies from the Movie Dialogue dataset for a training set of 1000 samples per feature)

```bash
python run.py experiment=finetune/movie_dialogue.yaml datamodule.feature_value=thriller datamodule.train_samples=1000
```