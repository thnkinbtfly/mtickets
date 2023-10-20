# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tb_log import add_to_summary_dict

from absl import flags
FLAGS = flags.FLAGS

add_target_per_lang_name = 'add_target_per_lang'
skip_vars = ['bias', 'mask', 'norm', 'Norm']

def check_name(tv_name):
    return all([skip_var not in tv_name for skip_var in skip_vars]) and (FLAGS.keep_var + '/') in tv_name


def create_optimizer(loss, init_lr: float, num_train_steps, num_warmup_steps, use_tpu,
                     total_train_steps=None,
                     language=None,
                     opt_weight_only=False):
  """Creates an optimizer training op."""
  language = tf.reshape(language, [-1])[0]
  if total_train_steps is None:
      total_train_steps = num_train_steps

  global_step = tf.compat.v1.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.compat.v1.train.polynomial_decay(
      learning_rate,
      global_step,
      total_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  add_to_summary_dict("lr", learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  exclude_from_decay = ["LayerNorm", "layer_norm", "bias"]
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=exclude_from_decay)

  if use_tpu:
    optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.compat.v1.trainable_variables()

  if opt_weight_only:
    tvars = [v for v in tvars if not "_mask" in v.name]

  grads = tf.gradients(ys=loss, xs=tvars)


  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  if FLAGS.accum_type == 'assign_grad':
      (accum_grads, accum_tvars) = zip(*[(g, v) for g, v in zip(grads, tvars) if check_name(v.name)])

      add_target_per_langs = [tf.compat.v1.get_variable(name=add_target_per_lang_name + str(i), shape=(FLAGS.total_languages,) + tuple(tv.shape),
                                                        initializer=tf.compat.v1.zeros_initializer,
                                                        trainable=False,
                                                        dtype=tv.dtype,
                                                        ) for i, tv in enumerate(accum_tvars)]

      add_targets = accum_grads
      update_op = tf.compat.v1.group(
          [tf.compat.v1.scatter_update(add_target_per_lang, [language], [add_target]) for (add_target_per_lang, add_target) in
           zip(add_target_per_langs, add_targets)])

      return tf.group(update_op, [global_step.assign(new_global_step)])

  elif FLAGS.accum_type == 'updates':
      train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=global_step)

      (accum_grads, accum_tvars) = zip(*[(g, v) for g, v in zip(grads, tvars) if check_name(v.name)])

      add_target_per_langs = [tf.compat.v1.get_variable(name=add_target_per_lang_name + str(i), shape=(FLAGS.total_languages,) + tuple(tv.shape),
                                                        initializer=tf.compat.v1.zeros_initializer,
                                                        trainable=False,
                                                        dtype=tv.dtype,
                                                        ) for i, tv in enumerate(accum_tvars)]

      add_targets = optimizer._opt.update_with_lrs if use_tpu else optimizer.update_with_lrs
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
      with tf.compat.v1.control_dependencies([train_op]):
          add_op = tf.compat.v1.group(
              [tf.compat.v1.scatter_nd_add(add_target_per_lang, [language], FLAGS.accum_weight * add_target) for (add_target_per_lang, add_target) in
               zip(add_target_per_langs, add_targets)])

      return add_op

  else:
      train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=global_step)
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
      return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    self.update_with_lrs = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.compat.v1.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())
      v = tf.compat.v1.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.compat.v1.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      if check_name(param_name):
        self.update_with_lrs.append(update_with_lr)

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
