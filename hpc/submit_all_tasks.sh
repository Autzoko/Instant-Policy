#!/bin/bash
# =============================================================
# submit_all_tasks.sh - Submit evaluation jobs for all RLBench
#                       tasks in parallel
#
# Usage:
#   bash hpc/submit_all_tasks.sh
# =============================================================

TASKS=(
    "plate_out"
    "open_box"
    "close_box"
    "toilet_seat_down"
    "toilet_seat_up"
    "toilet_roll_off"
    "close_microwave"
    "open_microwave"
    "phone_on_base"
    "push_button"
    "lift_lid"
    "slide_block"
    "basketball"
    "lamp_on"
    "put_rubbish"
    "umbrella_out"
    "buzz"
)

NUM_DEMOS="${NUM_DEMOS:-2}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-10}"

echo "Submitting ${#TASKS[@]} tasks (demos=$NUM_DEMOS, rollouts=$NUM_ROLLOUTS)"
echo "========================================================"

for task in "${TASKS[@]}"; do
    JOB_ID=$(TASK_NAME="$task" NUM_DEMOS="$NUM_DEMOS" NUM_ROLLOUTS="$NUM_ROLLOUTS" \
        sbatch --job-name="ip-${task}" \
               --export=ALL,TASK_NAME="$task",NUM_DEMOS="$NUM_DEMOS",NUM_ROLLOUTS="$NUM_ROLLOUTS" \
               hpc/submit_single.sbatch \
        | awk '{print $4}')
    echo "  Submitted: ${task}  (Job ID: ${JOB_ID})"
done

echo "========================================================"
echo "Done. Monitor with: squeue --me"
