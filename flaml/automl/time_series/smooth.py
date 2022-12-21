from typing import Sequence
import copy

import numpy as np


def moving_window_smooth(
    x: np.ndarray, win_size: int, match_length: bool = False
) -> np.ndarray:
    """
    Compute a moving window smooth,
    :param x: array, time is first dimension
    :param win_size: window size
    :param match_length: if true, pad the first values with the average over first win_size inputs
    :return: Smooth over a moving window
    """
    cumx = np.nancumsum(x, axis=0)
    out = (
        cumx[(win_size - 1) :]
        - np.concatenate([np.zeros((1, *cumx.shape[1:])), cumx[:-win_size]])
    ) / win_size
    if match_length:
        # padding = x[:win_size].mean(axis=0, keepdims=True).repeat(win_size - 1, axis=0)
        padding = np.nancumsum(x[: (win_size - 1)], axis=0)
        # this breaks causality, but prevents us losing info on the first point
        padding[0] += x[1]
        padding[0] /= 2.0
        for i in range(1, win_size - 1):
            padding[i] /= i + 1
        out = np.concatenate([padding, out])
        assert out.shape == x.shape
    return out


def nanexpsmooth(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.array(x)
    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        if np.isnan(out[i - 1]):
            out[i] = x[i]
        elif np.isnan(x[i]):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out


def expsmooth(x: Sequence, alpha: float, jump_from_zero: bool = False) -> Sequence:
    try:
        # put the import here so the function also works for environments without pytorch
        import torch

        if isinstance(x, torch.Tensor):
            # have to do it differently to not do in-place operations that screw up autograd
            out = [None] * len(x)
            out[0] = x[0]
            for i in range(1, len(x)):
                out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
                if jump_from_zero:
                    past_zeros = out[i - 1].abs() < 1e-6
                    out[i][past_zeros] = x[i][past_zeros]
            out = torch.stack(out)
            return out
        else:
            out = copy.deepcopy(x)
    except:
        out = copy.deepcopy(x)

    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
        if jump_from_zero:
            np.abs(past_zeros=out[i - 1]) < 1e-6
            out[i][past_zeros] = x[i][past_zeros]
    return out


def expsmooth_by_week(
    x: Sequence, alpha: float, jump_from_zero: bool = False
) -> Sequence:
    """
    Smooth things separately for each doy of the week
    :param x: a Sequence supporting arithmetical operations, with date as first dimension
    :return: a Sequence with each day of the week expmoothed separately
    """
    try:
        # put the import here so the function also works for environments without pytorch
        import torch

        if isinstance(x, torch.Tensor):
            out = torch.zeros_like(x)
        else:
            out = copy.deepcopy(x)
    except:
        out = copy.deepcopy(x)
    for wd in range(7):
        out[wd::7] = expsmooth(x[wd::7], alpha, jump_from_zero)
    return out


if __name__ == "__main__":
    I = np.eye(15)
    step = 5
    points = 38
    out = moving_window_smooth(I, step, True)
