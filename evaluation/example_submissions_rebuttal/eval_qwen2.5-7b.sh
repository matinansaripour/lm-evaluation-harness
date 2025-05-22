# Make sure to set your WANDB_API_KEY.
export HF_TOKEN=
export WANDB_API_KEY=
export HF_AUTH_TOKEN=
export HF_API_KEY=
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=Qwen/Qwen2.5-7B
export TOKENIZER=$MODEL
export NAME=Qwen2.5-7B-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--size 8 --wandb-entity matinansaripour --wandb-project robots-txt-rebuttal --wandb-id $NAME --bs 64 --consumed-tokens 18000000000000 --tasks scripts/evaluation/robots_eval2"

bash scripts/evaluation/submit_evaluation_robots.sh $MODEL $ARGS
