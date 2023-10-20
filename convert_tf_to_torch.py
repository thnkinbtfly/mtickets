"""
Convert BERT checkpoint.
copied from and updated
https://github.com/huggingface/transformers/blob/acc3bd9d2a73fcc7d3509767d65b2f40962d9330/src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py
"""
import os
import argparse
import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)

from transformers.models.bert.modeling_bert import BertConfig, BertForPreTraining
from util.bert_with_mask import BertForPreTrainingWithMask, BertConfigWithMask


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        if any(
                skip_word in name for skip_word in ["sum_grads", "control"]
        ):
            continue
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
                n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step",
                      "Adam", "Adam_1", "control", "beta1_power", "beta2_power"]
                for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    if 'layer_normalization' in scope_names[0]:
                        pointer = getattr(pointer, 'LayerNorm')
                    else:
                        pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.swapaxes(array, -2, -1)
            # array = np.transpose(array)
        try:
            assert (
                    pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path=None,
                                     with_mask=True):
    # Initialise PyTorch model
    config_cls = BertConfig if not with_mask else BertConfigWithMask
    model_cls = BertForPreTraining if not with_mask else BertForPreTrainingWithMask

    config = config_cls.from_json_file(bert_config_file)

    print(f"Building PyTorch model from configuration: {config}")
    model = model_cls(config)

    # Load weights from tf checkpoint
    model = load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    if pytorch_dump_path is not None:
        print(f"Save PyTorch model to {pytorch_dump_path}")
        torch.save(model.state_dict(), pytorch_dump_path)
        from util.io_utils import get_folderpath
        config.save_pretrained(get_folderpath(pytorch_dump_path))
    else:
        return model


import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
             "This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_output", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--with_mask", default=False, action='store_true', help="Path to the config, vocab folder."
    )
    parser.add_argument(
        "--hf_config_path", default='mbert', type=str, help="Path to the config, vocab folder."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint, args.config, args.pytorch_dump_output,
                                     args.with_mask)
    if args.hf_config_path:
        save_folder = os.path.dirname(args.pytorch_dump_output)
        for filename in ["tokenizer_config.json", "vocab.txt"]:
            shutil.copy(os.path.join(args.hf_config_path, filename), os.path.join(save_folder, filename))
