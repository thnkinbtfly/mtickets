import collections

import time
import numpy as np
import os, sys
from common import lang2id
from manipulate_ckpt import CkptChanger
from util.io_utils import get_ckpt_old_to_new
from optimization import add_target_per_lang_name
from util.other_utils import prev_curr

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
    os.makedirs(f"{ramdisk_dir}/{args.output_dir}", exist_ok=True)
    for suffix in file_suffixes:
        if not os.path.exists(f"{ramdisk_dir}/{args.output_dir}/{os.path.basename(ckpt + suffix)}"):
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
        if name not in done_set:
            agg_dict[name].append(np.reshape(log_name_to_val[log_name], (len(lang2id), -1)))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--ckpt_dirs', nargs='+', type=str)
    parser.add_argument('--start_index', default=0, type=int)
    parser.add_argument('--end_index', default=None, type=int)
    parser.add_argument('--save_sum_cossim', default=False, action='store_true')
    parser.add_argument('--save_cossim', default=False, action='store_true')
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
    for ckpt_dir in args.ckpt_dirs:
        ckpts = get_ckpt_old_to_new(ckpt_dir)
        end_index = len(ckpts) if args.end_index is None else args.end_index
        ckpts = ckpts[args.start_index:end_index]
        download_ckpt(ckpts[0])
        for curr_ckpt, next_ckpt in prev_curr(ckpts):
            download_ckpt(next_ckpt)
            load_gradients(curr_ckpt, save_cossim=args.save_cossim)
        load_gradients(ckpts[-1], save_cossim=args.save_cossim)

