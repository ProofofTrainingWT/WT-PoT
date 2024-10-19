import subprocess
import time

python_file = "data_processing\\k-s_test_verification.py"

delay_seconds = 20

log_index_list = list(range(1, 2))
pretrained_names = ['densenet121']

#reference 1
reference_index1 = 2
reference_name1 = 'resnet18'

for log_index in log_index_list:
    for model_name in pretrained_names:
        log_name = f'["{reference_name1}({reference_index1})","{model_name}({log_index})"]'
        subprocess.call(["python", python_file, "--log_name", log_name])

    time.sleep(delay_seconds)
