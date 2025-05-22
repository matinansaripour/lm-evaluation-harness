# Make sure to set your WANDB_API_KEY.
export HF_TOKEN=
export WANDB_API_KEY=
export HF_AUTH_TOKEN=
export HF_API_KEY=
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=Qwen/Qwen2.5-3B
export TOKENIZER=$MODEL
export NAME=Qwen2.5-3B-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--size 3 --wandb-entity matinansaripour --wandb-project robots-txt-rebuttal --wandb-id $NAME --bs 64 --consumed-tokens 18000000000000 --tasks scripts/evaluation/robots_eval"

bash scripts/evaluation/submit_evaluation_robots.sh $MODEL $ARGS
