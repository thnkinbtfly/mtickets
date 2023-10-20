import collections
import os
import pickle
from common import lang2id
import torch
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from util.bert_with_mask import BertForPreTrainingWithMask, BertConfigWithMask

@dataclass
class Args:
    snips_folder: str = field(default=None)
    model_name_or_path: str = field(default=None)
    output_dir: str = field(default=None)
    remain_ratio: float = field(default=0.75)
    use_random_snips: bool = field(default=False)



def get_mask_type(name):
    # regex expression to capture "xxx_mask" between dots
    # e.g. "bert.encoder.layer.0.attention.self.query.weight_mask.0"
    # will be captured as "weight_mask"
    import re
    assert name.endswith("_mask")
    layer_name = name.split('.')[-1]
    return layer_name

def normalize_snips(tmp_result_dict):
    for name, snip_val in tmp_result_dict.items():
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(snip_val, exponent).sum(), 1 / exponent)
        tmp_result_dict[name] /= norm_by_layer + 1e-20
    return tmp_result_dict


def sort_then_prune_snips(tmp_result_dict, remain_ratio):
    result_dict = {}
    for name, snip_val in tmp_result_dict.items():
        for i, v in enumerate(snip_val):
            result_dict[f"{name}.{i}"] = float(v)
    # else:
    #     assert do_norm is None
    snip_tuples = list(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    snip_tuples = snip_tuples[:int(len(snip_tuples) * remain_ratio)]
    return collections.OrderedDict(snip_tuples)

def update_to_total_snips(name_to_snips, total_snips, lang_id):
    for name, val in name_to_snips.items():
        total_snips[f"{name}.{lang_id}"] = val

def agg_snips_with_langsim(name_to_snips, langsim_list, method):
    if method == 'weightsum':
        total_snips = collections.defaultdict(lambda: 0)
        for lang_id, langsim in enumerate(langsim_list):
            for name, val in name_to_snips.items():
                total_snips[f"{name}.{lang_id}"] += val * langsim

def check_not_out_or_layer(k):
    return all([s not in k for s in ['output_mask', 'layer_mask']])


if __name__ == '__main__':
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    assert isinstance(args, Args)
    total_snips = {}
    for lang, lang_id in lang2id.items():
        lang_snips = pickle.load(open(f"{args.snips_folder}/{lang}.snip", "rb"))
        lang_snips = {k:v for k,v in lang_snips.items() if check_not_out_or_layer(k)}
        if args.use_random_snips:
            lang_snips = {k:torch.rand_like(v) for k,v in lang_snips.items()}

        lang_snips = normalize_snips(lang_snips, do_norm=args.norm_type)
        update_to_total_snips(lang_snips, total_snips, lang_id)

    total_snips = sort_then_prune_snips(total_snips, args.remain_ratio)


    config = BertConfigWithMask.from_pretrained(args.model_name_or_path)
    model = BertForPreTrainingWithMask.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    for name, param in model.named_parameters():
        if '_mask' in name:
            if not check_not_out_or_layer(name):
                param.data.fill_(1)

            else:
                for lang_id, mask_per_lang in enumerate(param):
                    for i, v in enumerate(mask_per_lang):
                        total_snip_name = f"{name}.{lang_id}.{i}"
                        if total_snip_name in total_snips: # remains
                            v.data.fill_(1)
                        else: # pruned
                            v.data.fill_(0)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)