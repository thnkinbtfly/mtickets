# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os

import numpy as np
import tensorflow.compat.v1 as tf

# from transformers import BertModel
from util.bert_with_mask import BertForPreTrainingWithMask, BertConfigWithMask


def convert_pytorch_checkpoint_to_tf(model: BertForPreTrainingWithMask, ckpt_dir: str, model_name: str):

    """
    Args:
        model: BertModel Pytorch model instance to be converted
        ckpt_dir: Tensorflow model directory
        model_name: model name

    Currently supported HF models:

        - Y BertModel
        - N BertForMaskedLM
        - Y BertForPreTrainingWithMask
        - N BertForMultipleChoice
        - N BertForNextSentencePrediction
        - N BertForSequenceClassification
        - N BertForQuestionAnswering
    """

    tensors_to_transpose = ("dense.weight", "attention.self.query", "attention.self.key", "attention.self.value", "fc1.weight", "fc2.weight")
    tensors_to_skip = ("decoder",)

    var_map = (
        ("layer.", "layer_"),
        ("word_embeddings.weight", "word_embeddings"),
        ("position_embeddings.weight", "position_embeddings"),
        ("token_type_embeddings.weight", "token_type_embeddings"),
        (".", "/"),
        ("LayerNorm/weight", "LayerNorm/gamma"),
        ("LayerNorm/bias", "LayerNorm/beta"),
        ("weight", "kernel"),
        ("seq_relationship/kernel", "seq_relationship/output_weights"), # originally seq_relationship/kernel, but changed because of previous...
        ("seq_relationship/bias", "seq_relationship/output_bias"),
        ("predictions/bias", "predictions/output_bias"),
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    prefix = '' if hasattr(model, 'bert') else 'bert/'
    def to_tf_var_name(name: str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return f"{prefix}{name}"

    def create_tf_var(tensor: np.ndarray, name: str, session: tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    tf.reset_default_graph()
    with tf.Session(config=config) as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            if any([x in var_name for x in tensors_to_skip]):
                print([x in var_name for x in tensors_to_skip])
                print(f"skipping {var_name}")
                continue
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]) and not '_mask' in var_name:
                # torch_tensor = torch_tensor.transpose(-2, -1)
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print(f"Successfully created {tf_name}: {np.allclose(tf_weight, torch_tensor)} {tf_weight.shape}")

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name))


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="model name e.g. bert-base-uncased")
    parser.add_argument("--tf_cache_dir", type=str, required=True, help="Directory in which to save tensorflow model")
    parser.add_argument("--ckpt_save_name", type=str, default=None, help="model name e.g. bert-base-uncased")
    parser.add_argument("--use_global_mask", default=False, action='store_true', help="model name e.g. bert-base-uncased")
    args = parser.parse_args(raw_args)

    bert_cls = BertForPreTrainingWithMask
    config = BertConfigWithMask.from_pretrained(args.model_name)
    model, loading_info = bert_cls.from_pretrained(args.model_name, output_loading_info=True, config=config)

    from util.io_utils import get_filename
    convert_pytorch_checkpoint_to_tf(model=model, ckpt_dir=args.tf_cache_dir, model_name=args.ckpt_save_name or get_filename(args.model_name, remove_ext=False).replace("-", "_") + ".ckpt")


if __name__ == "__main__":
    main()
