# import logging
# from functools import wraps
# from contextlib import contextmanager
# import torch


# @contextmanager
# def _ignore_torch_cuda_oom():
#     """
#     A context which ignores CUDA OOM exception from pytorch.
#     """
#     try:
#         yield
#     except RuntimeError as e:
#         # NOTE: the string may change?
#         if "CUDA out of memory. " in str(e):
#             pass
#         else:
#             raise


# def retry_if_gpu_oom(func):
#     """
#     Makes a function retry itself after encountering
#     pytorch's CUDA OOM error.
#     It will first retry after calling `torch.cuda.empty_cache()`.

#     If that still fails, it will then retry by trying to convert inputs to CPUs.
#     In this case, it expects the function to dispatch to CPU implementation.
#     The return values may become CPU tensors as well and it's user's
#     responsibility to convert it back to CUDA tensor if needed.

#     Args:
#         func: a stateless callable that takes tensor-like objects as arguments

#     Returns:
#         a callable which retries `func` if OOM is encountered.

#     Examples:
#     ::
#         output = retry_if_gpu_oom(some_torch_function)(input1, input2)
#         # output may be on CPU even if inputs are on GPU

#     Note:
#         1. When converting inputs to CPU, it will only look at each argument and check
#            if it has `.device` and `.to` for conversion. Nested structures of tensors
#            are not supported.

#         2. Since the function might be called more than once, it has to be
#            stateless.
#     """

#     def maybe_to_cpu(x):
#         try:
#             like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
#         except AttributeError:
#             like_gpu_tensor = False
#         if like_gpu_tensor:
#             return x.to(device="cpu")
#         else:
#             return x

#     @wraps(func)
#     def wrapped(*args, **kwargs):
#         with _ignore_torch_cuda_oom():
#             return func(*args, **kwargs)

#         # Clear cache and retry
#         torch.cuda.empty_cache()
#         with _ignore_torch_cuda_oom():
#             return func(*args, **kwargs)

#         # Try on CPU. This slows down the code significantly, therefore print a notice.
#         logger = logging.getLogger(__name__)
#         logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
#         new_args = (maybe_to_cpu(x) for x in args)
#         new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
#         return func(*new_args, **new_kwargs)

#     return wrapped


import logging
from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def _ignore_torch_gpu_oom():
  """
  A context manager that ignores CUDA and MPS OOM exceptions from PyTorch.
  """
  try:
    yield
  except RuntimeError as e:
    # Check for OOM error messages from CUDA and MPS
    error_str = str(e)
    if ("CUDA out of memory." in error_str) or ("MPS backend out of memory" in error_str):
      pass
    else:
      raise


def retry_if_gpu_oom(func):
  """
  Makes a function retry itself after encountering PyTorch's GPU OOM error (CUDA or MPS).
  It will first retry after calling `torch.cuda.empty_cache()` or `torch.mps.empty_cache()`.

  If that still fails, it will then retry by converting inputs to CPUs.
  In this case, it expects the function to dispatch to CPU implementation.
  The return values may become CPU tensors as well, and it's the user's
  responsibility to convert them back to the GPU if needed.

  Args:
      func: a stateless callable that takes tensor-like objects as arguments

  Returns:
      A callable which retries `func` if OOM is encountered.

  Examples:
      output = retry_if_gpu_oom(some_torch_function)(input1, input2)
      # output may be on CPU even if inputs are on GPU

  Notes:
      1. When converting inputs to CPU, it will only look at each argument and check
         if it has `.device` and `.to` for conversion. Nested structures of tensors
         are not supported.

      2. Since the function might be called more than once, it has to be stateless.
  """

  def maybe_to_cpu(x):
    try:
      is_gpu_tensor = x.device.type in ("cuda", "mps") and hasattr(x, "to")
    except AttributeError:
      is_gpu_tensor = False
    if is_gpu_tensor:
      return x.to(device="cpu")
    else:
      return x

  @wraps(func)
  def wrapped(*args, **kwargs):
    with _ignore_torch_gpu_oom():
      return func(*args, **kwargs)

    # Clear CUDA cache and retry
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
      torch.mps.empty_cache()

    with _ignore_torch_gpu_oom():
      return func(*args, **kwargs)

    # Try on CPU. This slows down the code significantly, so print a notice.
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to copy inputs of {func} to CPU due to GPU OOM")
    new_args = (maybe_to_cpu(x) for x in args)
    new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
    return func(*new_args, **new_kwargs)

  return wrapped
