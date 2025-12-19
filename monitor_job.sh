#!/bin/bash

# 1. Submit the job and capture the output
# expected output: "Submitted batch job 123456"
OUTPUT=$(sbatch job_launcher.sh)
echo "$OUTPUT"

# 2. Extract Job ID
JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')

if [[ -z "$JOB_ID" ]]; then
    echo "Error: Could not extract Job ID. Submission might have failed."
    exit 1
fi

LOG_FILE="logs/job_${JOB_ID}.out"
ERR_FILE="logs/job_${JOB_ID}.err"

echo "---------------------------------------------------"
echo "Job $JOB_ID is queued."
echo "Waiting for logs to appear (this starts when the job enters Running state)..."
echo "---------------------------------------------------"

# 3. Wait for the log file to be created (polling)
while [ ! -f "$LOG_FILE" ]; do
    # Check if job crashed early (only err file exists)
    if [ -f "$ERR_FILE" ] && [ ! -f "$LOG_FILE" ]; then
        echo "Alert: Error file found but no output file. Job might have crashed."
        echo "Dumping Error Log:"
        cat "$ERR_FILE"
        exit 1
    fi
    sleep 2
done

# 4. Stream the log in the background
echo "Log file created! Streaming output..."
echo "---------------------------------------------------"
tail -f "$LOG_FILE" &
TAIL_PID=$!

# 5. Monitor Job Status to stop tailing when job ends
echo "Monitoring job status..."
while true; do
    # Check if job is still in squeue using line count (more robust)
    # squeue -h means no header. If job is gone, output is empty.
    if [ $(squeue -h -j "$JOB_ID" 2>/dev/null | wc -l) -eq 0 ]; then
        echo ""
        echo "---------------------------------------------------"
        echo "Job $JOB_ID has finished. Stopping log stream."
        # Wait a bit to let final logs flush
        sleep 5
        # Kill the tail process
        kill $TAIL_PID 2>/dev/null
        break
    fi
    sleep 2
done

# Check for error again just in case
if [ -s "$ERR_FILE" ]; then
    echo "dumping error log content (if any):"
    cat "$ERR_FILE"
fi
