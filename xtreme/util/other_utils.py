import importlib.util
import inspect
import itertools
from collections import OrderedDict
from dataclasses import dataclass
from itertools import tee
from typing import List, Any, Dict
import torch


def is_torch_tpu_available():
    # copied from https://github.com/huggingface/transformers/blob/024cd19bb7c188a0e4aa681d248ad9f47587ddab/src/transformers/file_utils.py#L280
    # if not _torch_available:
    #     return False
    # This test is probably enough, but just in case, we unpack a bit.
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if importlib.util.find_spec("torch_xla.core") is None:
        return False
    return importlib.util.find_spec("torch_xla.core.xla_model") is not None


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm


def wait_for_everyone(msg="accelerate.utils.wait_for_everyone"):
    from accelerate.state import AcceleratorState, DistributedType
    # copied from https://github.com/huggingface/accelerate/blob/f1333b54ad0141d162de6e6f04893ff7fd1f7e36/src/accelerate/utils.py#L252
    # Modified because series of xm.rendezvous with same msg seemed to be synced in different lines...
    if AcceleratorState().distributed_type == DistributedType.MULTI_GPU:
        torch.distributed.barrier()
    elif AcceleratorState().distributed_type == DistributedType.TPU:
        xm.rendezvous(msg)


def get_total_batch_size(per_dev_batch_size, num_procs, grad_acc_steps=1):
    """Set grad_acc_steps=1 when eval. used only in train"""
    return per_dev_batch_size * num_procs * grad_acc_steps


def cat_lists(lists):
    return list(itertools.chain.from_iterable(lists))


def str_to_list(space_splitted_str, type=float):
    space_splitted_str = str(space_splitted_str)
    return [type(o) for o in space_splitted_str.split()]


def l_of_dic_to_dic_of_l(l_of_dic: List[dict]):
    result = OrderedDict([(k, []) for k in l_of_dic[0].keys()])
    for dic in l_of_dic:
        for k, v in dic.items():
            assert k in result
            result[k].append(v)
    return result


def prev_curr(iterable):
    # from https://docs.python.org/3/library/itertools.html#itertools-recipes
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def get_kwargs():
    """
    Gets kwargs of given called function.
    You can use this instead "local()"
    got idea from https://stackoverflow.com/questions/582056/getting-list-of-parameter-names-inside-python-function
    """
    frame = inspect.stack()[1][0]
    varnames, _, _, values = inspect.getargvalues(frame)

    called_from_class_method = (varnames[0] == 'self')
    if called_from_class_method:
        varnames = varnames[1:]

    kwargs = {i: values[i] for i in varnames}
    return kwargs

