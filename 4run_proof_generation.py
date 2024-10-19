import subprocess
import time

python_file = "generate_wt_proof.py"

log_index_list = list(range(1, 2))
delay_seconds = 20

pretrained_names = ['densenet121']

for log_index in log_index_list:
    for model_name in pretrained_names:
        subprocess.call(["python", python_file, "--log_index", str(log_index), "--pretrained_name", model_name, "--wm_classes", '2'])
    time.sleep(delay_seconds)
