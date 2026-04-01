import json
import time

def log_usage(n_samples):
    log = {
        "timestamp": time.time(),
        "samples_generated": n_samples
    }

    with open("usage_log.json", "a") as f:
        f.write(json.dumps(log) + "\n")