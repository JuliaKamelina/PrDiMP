import numpy as np
import torch

def fft2(x):
    return np.fft.fft(np.fft.fft(x, axis=1), axis=0).astype(np.complex64)

def ifft2(x):
    return np.fft.ifft(np.fft.ifft(x, axis=1), axis=0).astype(np.complex64)

def cfft2(x):
    in_shape = x.shape
    # if both dimensions are odd
    if in_shape[0] % 2 == 1 and in_shape[1] % 2 == 1:
        xf = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
    else:
        out_shape = list(in_shape)
        out_shape[0] =  in_shape[0] + (in_shape[0] + 1) % 2
        out_shape[1] =  in_shape[1] + (in_shape[1] + 1) % 2
        out_shape = tuple(out_shape)
        xf = np.zeros(out_shape, dtype=np.complex64)
        xf[:in_shape[0], :in_shape[1]] = np.fft.fftshift(np.fft.fftshift(fft2(x), 0), 1).astype(np.complex64)
        if out_shape[0] != in_shape[0]:
            xf[-1,:] = np.conj(xf[0,::-1])
        if out_shape[1] != in_shape[1]:
            xf[:,-1] = np.conj(xf[::-1,0])
    return xf

def cifft2(xf):
    x = np.real(ifft2(np.fft.ifftshift(np.fft.ifftshift(xf, 0),1))).astype(np.float32)
    return x

def interpolate_dft(xf, interp1_fs, interp2_fs):
    return [xf_ * interp1_fs_ * interp2_fs_
            for xf_, interp1_fs_, interp2_fs_ in zip(xf, interp1_fs, interp2_fs)]

def compact_fourier_coeff(xf):
    if isinstance(xf, list):
        return [x[:, :(x.shape[1]+1)//2, :] for x in xf]
    else:
        return xf[:, :(xf.shape[1]+1)//2, :]

def shift_sample(xf, shift, kx, ky):
    shift_exp_y = [np.exp(1j * shift[0] * y).astype(np.complex64) for y in ky]
    shift_exp_x = [np.exp(1j * shift[1] * x).astype(np.complex64) for x in kx]
    xf = [xf_ * sy_.reshape(-1, 1, 1, 1) * sx_.reshape((1, -1, 1, 1))
            for xf_, sy_, sx_ in zip(xf, shift_exp_y, shift_exp_x)]
    return xf

def symmetrize_filter(hf):
    for i in range(len(hf)):
        dc_ind = int((hf[i].shape[0]+1) / 2)
        hf[i][dc_ind:, -1, :] = np.conj(np.flipud(hf[i][:dc_ind-1, -1, :]))
    return hf

def full_fourier_coeff(xf):
    """
        reconstruct full fourier series
    """

    xf = [np.concatenate([xf_, np.conj(np.rot90(xf_[:, :-1,:], 2))], axis=1) for xf_ in xf]
    return xf

def sample_fs(xf, grid_sz=None):
    """
        Samples the Fourier series
    """

    sz = xf.shape[:2]
    if grid_sz is None or sz == grid_sz:
        x = sz[0] * sz[1] * cifft2(xf)
    else:
        sz = np.array(sz)
        grid_sz = np.array(grid_sz)
        if np.any(grid_sz < sz):
            raise("The grid size must be larger than or equal to the siganl size")

        tot_pad = grid_sz - sz
        pad_sz = np.ceil(tot_pad / 2).astype(np.int32)
        xf_pad = np.pad(xf, tuple(pad_sz), 'constant')
        if np.any(tot_pad % 2 == 1):
            # odd padding
            xf_pad = xf_pad[:xf_pad.shape[0]-(tot_pad[0] % 2), :xf_pad.shape[1]-(tot_pad[1] % 2)]
        x = grid_sz[0] * grid_sz[1] * cifft2(xf_pad)
    return x

def resize_DFT(inputdft, desired_len):
    input_len = len(inputdft)
    minsz = min(input_len, desired_len)

    scaling = desired_len / input_len

    resize_dft = np.zeros(desired_len, dtype=inputdft.dtype)

    mids = int(np.ceil(minsz / 2))
    mide = int(np.floor((minsz - 1) / 2))

    resize_dft[:mids] = scaling * inputdft[:mids]
    resize_dft[-mide:] = scaling * inputdft[-mide:]
    return resize_dft

def max2d(a: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Computes maximum and argmax in the last two dimensions."""

    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax