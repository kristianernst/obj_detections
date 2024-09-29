# Copyright (c) Facebook, Inc. and its affiliates.
import functools

import numpy as np
import torch
import torch.distributed as dist

try:
  import cloudpickle
except ImportError:
  # cloudpickle is an optional dependency at the moment
  pass


def get_world_size() -> int:
  if not dist.is_available():
    return 1
  if not dist.is_initialized():
    return 1
  return dist.get_world_size()


def get_rank() -> int:
  if not dist.is_available():
    return 0
  if not dist.is_initialized():
    return 0
  return dist.get_rank()


@functools.lru_cache()
def _get_global_gloo_group():
  """
  Return a process group based on gloo backend, containing all the ranks
  The result is cached.
  """
  if dist.get_backend() == "nccl":
    return dist.new_group(backend="gloo")
  else:
    return dist.group.WORLD


def all_gather(data, group=None):
  """
  Run all_gather on arbitrary picklable data (not necessarily tensors).

  Args:
      data: any picklable object
      group: a torch process group. By default, will use a group which
          contains all ranks on gloo backend.

  Returns:
      list[data]: list of data gathered from each rank
  """
  if get_world_size() == 1:
    return [data]
  if group is None:
    group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
  world_size = dist.get_world_size(group)
  if world_size == 1:
    return [data]

  output = [None for _ in range(world_size)]
  dist.all_gather_object(output, data, group=group)
  return output


def gather(data, dst=0, group=None):
  """
  Run gather on arbitrary picklable data (not necessarily tensors).

  Args:
      data: any picklable object
      dst (int): destination rank
      group: a torch process group. By default, will use a group which
          contains all ranks on gloo backend.

  Returns:
      list[data]: on dst, a list of data gathered from each rank. Otherwise,
          an empty list.
  """
  if get_world_size() == 1:
    return [data]
  if group is None:
    group = _get_global_gloo_group()
  world_size = dist.get_world_size(group=group)
  if world_size == 1:
    return [data]
  rank = dist.get_rank(group=group)

  if rank == dst:
    output = [None for _ in range(world_size)]
    dist.gather_object(data, output, dst=dst, group=group)
    return output
  else:
    dist.gather_object(data, None, dst=dst, group=group)
    return []


def is_main_process() -> bool:
  return get_rank() == 0


def synchronize():
  """
  Helper function to synchronize (barrier) among all processes when
  using distributed training
  """
  if not dist.is_available():
    return
  if not dist.is_initialized():
    return
  world_size = dist.get_world_size()
  if world_size == 1:
    return
  if dist.get_backend() == dist.Backend.NCCL:
    # This argument is needed to avoid warnings.
    # It's valid only for NCCL backend.
    dist.barrier(device_ids=[torch.cuda.current_device()])
  else:
    dist.barrier()


def shared_random_seed():
  """
  Returns:
      int: a random number that is the same across all workers.
      If workers need a shared RNG, they can use this shared seed to
      create one.

  All workers must call this function, otherwise it will deadlock.
  """
  ints = np.random.randint(2**31)
  all_ints = all_gather(ints)
  return all_ints[0]


class PicklableWrapper:
  """
  Wrap an object to make it more picklable, note that it uses
  heavy weight serialization libraries that are slower than pickle.
  It's best to use it only on closures (which are usually not picklable).

  This is a simplified version of
  https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
  """

  def __init__(self, obj):
    while isinstance(obj, PicklableWrapper):
      # Wrapping an object twice is no-op
      obj = obj._obj
    self._obj = obj

  def __reduce__(self):
    s = cloudpickle.dumps(self._obj)
    return cloudpickle.loads, (s,)

  def __call__(self, *args, **kwargs):
    return self._obj(*args, **kwargs)

  def __getattr__(self, attr):
    # Ensure that the wrapped object can be used seamlessly as the previous object.
    if attr not in ["_obj"]:
      return getattr(self._obj, attr)
    return getattr(self, attr)
