import os
import subprocess

start = 400
end = 600
exp_name = "MLT2017"
gpu_id = "1"
tr_thresh = 0.6
tcl_thresh = 0.4
expend = 0.255

if __name__ == "__main__":

    for epoch in range(start, end+1, 35):
        try:
            subprocess.call(['python', 'eval_textsnake.py',
                         "--exp_name", exp_name, "--checkepoch", str(epoch), '--gpu', gpu_id])
        except:
            continue

