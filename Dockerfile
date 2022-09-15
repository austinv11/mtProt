# Based on: https://github.com/Lightning-AI/lightning/tree/master/dockers
FROM pytorchlightning/pytorch_lightning:base-conda-py3.9-torch1.12-cuda11.3.1

RUN conda install --yes -c conda-forge ncurses

# Install dependencies
ADD . .
RUN conda install --yes --freeze-installed -c conda-forge --file requirements.txt

RUN wandb login $(cat wandb_token.txt)

RUN wandb sweep sweep.yaml |& grep "wandb agent " | cut -d" " -f8 > wandb_agent_id.txt

ENTRYPOINT wandb agent $(cat wandb_agent_id.txt)