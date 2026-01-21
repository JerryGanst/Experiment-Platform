#!/bin/bash
# Monitor HACE experiment progress
# Usage: ssh -p 23 root@117.50.34.209 "bash /cloud/cloud-ssd1/Experiment-Platform/scripts/monitor_experiment.sh"

cd /cloud/cloud-ssd1/Experiment-Platform

echo "=========================================="
echo "HACE Experiment Monitor"
echo "=========================================="

# Check running processes
echo -e "\n[Running Processes]"
ps aux | grep run_hace | grep -v grep

# Check GPU usage
echo -e "\n[GPU Status]"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

# Check progress for each experiment
echo -e "\n[Experiment Progress]"
for dir in results_clean/hace_*/; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        count=$(wc -l < "${dir}qmsum.jsonl" 2>/dev/null || echo "0")
        echo "  $name: $count/200 samples"
    fi
done

# Check latest log
echo -e "\n[Latest Log (last 5 lines)]"
if [ -f "logs/reverse_disp_run.log" ]; then
    echo "=== reverse_disp_run.log ==="
    tail -5 logs/reverse_disp_run.log | grep -v "^tensor"
fi
if [ -f "logs/normal_run.log" ]; then
    echo "=== normal_run.log ==="
    tail -5 logs/normal_run.log | grep -v "^tensor"
fi

echo -e "\n=========================================="
