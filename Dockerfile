# Based on: https://github.com/Lightning-AI/lightning/tree/master/dockers
FROM pytorchlightning/pytorch_lightning:base-conda-py3.9-torch1.12-cuda11.3.1

RUN conda update -n base -c defaults conda

RUN conda install --yes -n base conda-libmamba-solver
RUN conda config --system --set solver libmamba

RUN conda install --yes -c conda-forge ncurses

# Mitigate DNS problems
RUN cat custom_hosts.txt >> /etc/hosts

RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

ARG sweep_file

# Install dependencies
ADD . .

RUN conda install --yes --freeze-installed -c conda-forge --file requirements.txt

RUN wandb login $(cat wandb_token.txt)

RUN wandb sweep $sweep_file |& grep "wandb agent " | cut -d" " -f8 > wandb_agent_id.txt

ENTRYPOINT wandb agent $(cat wandb_agent_id.txt)