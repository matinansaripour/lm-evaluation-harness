# Make sure to set your WANDB_API_KEY.
export HF_TOKEN=
export WANDB_API_KEY=
export HF_AUTH_TOKEN=
export HF_API_KEY=
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=/capstor/scratch/cscs/dfan/ckpts/hf-checkpoints/robots-txt/robots-filtered
export NAME=robots-filtered-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--size 1 --wandb-entity matinansaripour --wandb-project robots-txt-rebuttal --wandb-id $NAME --bs 64 --tokens-per-iter 2064384 --tasks scripts/evaluation/robots_eval"

bash scripts/evaluation/submit_evaluation_robots.sh $MODEL $ARGS --iterations "48441"
