# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered/checkpoints/ 
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top1-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top5-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top10-domains/checkpoints/
export MODEL=/iopsstor/scratch/cscs/ahgele/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1-cooldown/apertus3-1b-21-nodes-cooldown-short/checkpoints
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt-1b-cooldown/apertus3-1b-21-nodes-100bt-MathSciDomains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt-1b-cooldown/apertus3-1b-21-nodes-100bt-NewsDomains/checkpoints/
export NAME=robots-filtered-Ori-cool-down-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--convert-to-hf --size 1 --wandb-entity matinansaripour --wandb-project robots-txt-cooldown --wandb-id $NAME --bs 64 --tokens-per-iter 2064384 --tasks scripts/evaluation/robots_eval"

bash scripts/evaluation/submit_evaluation_robots.sh $MODEL $ARGS --iterations "850000"
