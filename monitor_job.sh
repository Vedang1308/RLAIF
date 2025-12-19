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

# 4. Stream the log
echo "Log file created! Streaming output..."
echo "---------------------------------------------------"
tail -f "$LOG_FILE"
