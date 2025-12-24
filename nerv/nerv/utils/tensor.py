import numpy as np

import torch
import torch.distributed as dist


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    assert isinstance(tensor, np.ndarray)
    return tensor


def batch_gather(tensor, idx):
    """Gather tensor by idx in batch. Only gather 1 position in the last dim.

    Args:
        tensor (torch.Tensor): [N_1, N_2, ..., N_n].
        idx (torch.Tensor): [N_1, ..., N_{k-1}].

    Returns:
        torch.Tensor: [N_1, ..., N_{k-1}, N_{k+1}, ..., N_n].
    """
    gather_dim = len(idx.shape)
    return batch_gather_k(tensor, idx.unsqueeze(-1)).squeeze(gather_dim)


def batch_gather_k(tensor, idx):
    """Gather tensor by idx in batch. The last dim is K positions to gather.

    Args:
        tensor (torch.Tensor): [N_1, N_2, ..., N_n].
        idx (torch.Tensor): [N_1, ..., N_{k-1}, M], M could be any value.

    Returns:
        torch.Tensor: [N_1, ..., N_{k-1}, M, N_{k+1}, ..., N_n].
    """
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(idx, (torch.LongTensor, torch.cuda.LongTensor))
    idx_shape = idx.shape
    k = len(idx_shape)

    # some easy solutions
    if k == 1:
        return tensor[idx]
    elif k == 2:
        batch_idx = torch.arange(tensor.shape[0])[:, None].type_as(idx)
        return tensor[batch_idx, idx]

    # the idea is to view the dims before last idx dim to form a new batch dim
    tensor_shape = tensor.shape
    n = len(tensor_shape)
    assert n >= k
    out_shape = list(idx.shape) + list(tensor_shape[k:])
    # view [B, N1, ..., N_(k-1)] to form a new batch dim
    view_shape = [-1] + list(tensor_shape[k - 1:])
    view_tensor = tensor.view(view_shape)
    view_idx = idx.view(-1, idx.shape[-1])
    return batch_gather_k(view_tensor, view_idx).view(out_shape)


def batch_cat_vec(tensor, value_vec, dim):
    """Concat some values at the end of a tensor along one dim.

    Useful in e.g. concat some indicator to different input data.

    Args:
        tensor (torch.Tensor): [N_1, N_2, ..., N_n].
        value_vec (torch.Tensor | List[Any]): a d-len vector to be concated.
        dim (int): specifies the dimention.

    Returns:
        torch.Tensor: [N_1, N_2, ... N_{dim} + d, ..., N_n].
    """
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(dim, int)
    tensor_shape = list(tensor.shape)
    n = len(tensor_shape)
    if dim < 0:
        dim = n + dim
    if not isinstance(value_vec, torch.Tensor):
        value_vec = torch.tensor(value_vec)
    value_vec = value_vec.type_as(tensor)
    # expand shape to match tensor
    for dim_ in range(n):
        if dim_ != dim:
            value_vec = value_vec.unsqueeze(dim=dim_)
    tensor_shape[-1] = 1
    value_vec = value_vec.repeat(tensor_shape)
    return torch.cat([tensor, value_vec], dim=dim)


def ddp_all_gather(tensor):
    """Warpper function for torch.distributed.all_gather().
    Automatically create an empty list for gathering and stack the result.

    Args:
        tensor (torch.Tensor): `shape`, should be on cuda.
            Can also be a scalar (shape [] tensor)!

    Returns:
        torch.Tensor: [world_size, `*shape`], gathered tensors from all GPUs.
    """
    output = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    output = torch.stack(output, dim=0)
    return output


class AllGatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                tensor: torch.Tensor,
                reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_dtype = grad_output.dtype
        input_list = [
            grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size())
        ]
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        dist.reduce_scatter(grad_input, input_list)
        return grad_input.to(grad_dtype)


def ddp_all_gather_w_grad(tensor):
    """All gather operation with gradient backward support."""
    return AllGatherFunction.apply(tensor)
