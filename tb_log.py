import six
import tensorflow.compat.v1 as tf
try:
    from tensorflow.contrib.tpu.python.ops.tpu_ops import cross_replica_sum
    from tensorflow.contrib.tpu.python.tpu import tpu_function
    from tensorflow.contrib.tpu import TPUEstimatorSpec, CrossShardOptimizer, bfloat16_scope
    import tensorflow.contrib.summary as tf2_summary
    all_summary_ops = tf2_summary.all_summary_ops
    always_record = tf2_summary.always_record_summaries
except ImportError:
    from tensorflow.python.tpu import tpu_function
    from tensorflow.compat.v1.tpu import cross_replica_sum
    from tensorflow.compat.v1.estimator.tpu import TPUEstimatorSpec
    from tensorflow.compat.v1.tpu import CrossShardOptimizer, bfloat16_scope
    import tensorflow.summary as tf2_summary
    from tensorflow.compat.v1.summary import all_v2_summary_ops as all_summary_ops
    always_record = lambda: tf2_summary.record_if(True)

try:
    from tensorflow.contrib.tpu import TPUEstimator, InputPipelineConfig, RunConfig, TPUConfig
except:
    from tensorflow.compat.v1.estimator.tpu import TPUEstimator, InputPipelineConfig, RunConfig, TPUConfig
# from compat_import import tf2_summary, all_summary_ops

summary_dict = {}


def add_to_summary_dict(name, tensor):
    global summary_dict
    summary_dict[name] = tensor

def get_summary_dict():
    global summary_dict
    return summary_dict


def build_host_call_for_tensorboard(save_dir, iterations_per_loop, log_on_tb_steps):
    global summary_dict
    def host_call_fn(**kwargs):
        """
        writes the {"tag_name": tensor} dict to tensorboard. got idea from
        https://github.com/tensorflow/tensor2tensor/blob/bf33311314005528482ea50b098d1aca8da85d84/tensor2tensor/utils/t2t_model.py#L2157
        """
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        gs = kwargs.pop("gs")[0]
        with tf2_summary.create_file_writer(save_dir, max_queue=iterations_per_loop).as_default():
            with tf2_summary.record_if(lambda: tf.math.equal(gs % log_on_tb_steps, 0)):
                for name, tensor in sorted(six.iteritems(kwargs)):
                    tf2_summary.scalar(name, tensor[0], step=gs)
                return all_summary_ops()

    for key, value in summary_dict.items():
        summary_dict[key] = tf.reshape(value, [1])

    host_call = (host_call_fn, summary_dict)

    return host_call
