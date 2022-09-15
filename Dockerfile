# Based on: https://github.com/Lightning-AI/lightning/tree/master/dockers
FROM pytorchlightning/pytorch_lightning:base-conda-py3.9-torch1.12-cuda11.3.1

# Install dependencies
ADD . .
RUN conda install --yes --freeze-installed -c conda-forge --file requirements.txt

RUN wandb login $(cat wandb_token.txt)

RUN wandb sweep sweep.yaml | cat test.txt | grep "wandb agent " | cut -d" " -f8 > wandb_agent_id.txt

CMD ['wandb', 'agent', '$(cat wandb_agent_id.txt)']