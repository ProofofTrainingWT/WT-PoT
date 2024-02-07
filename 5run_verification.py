import subprocess
import time

# 要重复运行的Python文件
# python_file = "plot_process_main(new_mec-statistic).py"
# python_file = "plot_process_main(new_mec-statistic)-sec.py"
python_file = "data_processing\\k-s_test_verification.py"

# 重复运行之间的延迟时间（秒）
delay_seconds = 20

log_index_list = list(range(2, 3))
pretrained_names = ['resnet18'] #, 'resnet50'

#reference 1
reference_index1 = 2
reference_name1 = 'resnet18'

for log_index in log_index_list:
    for model_name in pretrained_names:
        log_name = f'["{reference_name1}({reference_index1})","{model_name}({log_index})"]'
        subprocess.call(["python", python_file, "--log_name", log_name])

    time.sleep(delay_seconds)

# log_index_list = list(range(1, 3))
# pretrained_names = ['resnet18'] #, 'resnet50'

# for log_index in log_index_list:
#     for model_name in pretrained_names:
#         log_name = f'["{reference_name1}({reference_index1})","{reference_name2}({reference_index2})","{model_name}({log_index})"]'
#         subprocess.call(["python", python_file, "--log_name", log_name])
#
#     time.sleep(delay_seconds)
