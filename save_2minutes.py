import subprocess
import time
from datetime import datetime, timedelta

# Path to the recording script
SCRIPT_PATH = "BLM_recording_script_SingleChannel.py"
CH_1 = "1 0.04 CH1"
CH_2 = "2 0.0012 CH2"
CH_3 = "3 0.002 CH3"

def run_script_for_duration(script_path, options, duration_seconds):
    try:
        # Start the subprocess
        process = subprocess.Popen(["python", script_path]+options.split())
        print(f"[{datetime.now()}] Started recording script with PID {process.pid}")
        
        # Wait for the specified duration
        time.sleep(duration_seconds)

        # Terminate the process
        process.terminate()
        try:
            process.wait(timeout=5)  # give it a moment to clean up
        except subprocess.TimeoutExpired:
            process.kill()  # force kill if it doesn't stop

        print(f"[{datetime.now()}] Stopped recording script.")

    except Exception as e:
        print(f"[{datetime.now()}] Error: {e}")

def wait_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    wait_time = (next_hour - now).total_seconds()
    print(f"[{now}] Waiting {int(wait_time)} seconds until next run.")
    time.sleep(wait_time)

if __name__ == "__main__":
    while True:
        run_script_for_duration(SCRIPT_PATH, CH_1, duration_seconds=120)  # 2 minutes
        run_script_for_duration(SCRIPT_PATH, CH_2, duration_seconds=120)  # 2 minutes
        run_script_for_duration(SCRIPT_PATH, CH_3, duration_seconds=120)  # 2 minutes
        #wait_until_next_hour()
