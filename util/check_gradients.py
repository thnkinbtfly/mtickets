import collections

import time
import numpy as np
import os, sys
from common import lang2id
from manipulate_ckpt import CkptChanger
from util.io_utils import get_ckpt_old_to_new
from optimization import add_target_per_lang_name

file_suffixes = [
    '.meta',
    '.index',
    '.data-00000-of-00001'
]
ramdisk_dir = f"/ramdisk"

def save_matrix_as_csv(ndarray, filename):
    import pandas as pd
    df = pd.DataFrame(ndarray)
    df.to_csv(filename)

def download_ckpt(ckpt):
    # os.system(f"gsutil -m cp -r {os.path.join(args.ckpt_dir, ckpt)}\* {args.output_dir}/ &")
    os.makedirs(f"{ramdisk_dir}/{args.output_dir}", exist_ok=True)
    for suffix in file_suffixes:
        os.system(f"rm {ramdisk_dir}/{args.output_dir}/{os.path.basename(ckpt + suffix)}* ")
        os.system(f"gsutil -m cp {ckpt}{suffix} {ramdisk_dir}/{args.output_dir}/ &")

def load_gradients(ckpt, save_cossim=False):
    save_dir = f"{args.output_dir}/{os.path.basename(ckpt)}"
    local_ckpt = f"{ramdisk_dir}/{save_dir}"

    while True:
        if not all([os.path.exists(f"{local_ckpt}{suffix}") for suffix in file_suffixes]):
            print(f"waiting for {ckpt}")
            time.sleep(5)
        else:
            break
        # time.sleep(20)
    print("loading...")
    changer = CkptChanger(local_ckpt)

    done_set = set()
    print("least items")
    for i, log_name in enumerate(log_names):
        accum_val = changer.get_val(f"{add_target_per_lang_name}{i}")
        log_name_to_accum_val[log_name] = accum_val

    agg_dict = collections.defaultdict(list)
    add_to_agg_dict(agg_dict, log_name_to_blank, done_set, log_name_to_accum_val)
    accum_val = np.concatenate(agg_dict["."], axis=1)

    if save_cossim:
        norms = np.linalg.norm(accum_val, axis=1, keepdims=True)
        accum_val = accum_val / (norms + 1e-12)
        cossim_matrix = np.matmul(accum_val, accum_val.T)
        save_matrix_as_csv(cossim_matrix, f"{save_dir}_heatmap_annot.csv")

    for suffix in file_suffixes:
        os.remove(f"{local_ckpt}{suffix}")
    return accum_val



def add_to_agg_dict(agg_dict,log_name_to_name_dict,  done_set, log_name_to_val):
    for log_name, name in log_name_to_name_dict.items():
        print(name, 'adding')
        if name not in done_set:
            agg_dict[name].append(np.reshape(log_name_to_val[log_name], (len(lang2id), -1)))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--train_steps', default=1000, type=int)
    parser.add_argument('--ckpt_dirs', nargs='+', type=str)
    args = parser.parse_args()

    log_names = [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense',
        'intermediate.dense',
        'output.dense',
    ]
    log_name_to_accum_val = {}
    log_name_to_blank = {k: '.' for k in log_names}

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/log_names.txt", "a") as f:
        for ckpt_dir in args.ckpt_dirs:
            ckpts = get_ckpt_old_to_new(ckpt_dir)
            orig_ckpt = ckpts[0]
            orig_steps = int(ckpts[0].split('-')[-1])
            new_ckpt = ckpts[0][:-(len(str(orig_steps)) + 1)] + f"-{orig_steps + args.train_steps}"
            download_ckpt(ckpts[0])
            download_ckpt(new_ckpt)
            grad_1 = load_gradients(ckpts[0])
            grad_2 = load_gradients(new_ckpt, save_cossim=True)
            diff = grad_1 - grad_2
            dup = np.where(np.all(np.isclose(diff, 0), axis=1))
            f.write(f"{ckpt_dir}\n{dup}\n")
