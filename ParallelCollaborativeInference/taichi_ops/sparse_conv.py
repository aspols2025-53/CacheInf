import torch
import numpy as np
import taichi as ti
from taichi import f32, i32

def wrap_shape(total, factor):
    return int(np.ceil(total / factor))

H_BLOCK = 16
W_BLOCK = 16
C1_BLOCK = 4
C2_BLOCK = 4

@ti.data_oriented
class SparseTensor:
    def __init__(self, shape, indices: torch.Tensor=None, data: torch.Tensor=None, x=None, active_uv=None):
        # indices: [N,2]
        # data: [N,C]
        self.shape = shape
        b,c,h,w = shape
        if x is None or active_uv is None:
            self.x = ti.field(ti.f32)
            block = ti.root.pointer(ti.ijkl, (b, wrap_shape(c,C1_BLOCK), 
                                              wrap_shape(h,H_BLOCK),
                                              wrap_shape(w,W_BLOCK)))
            pixel = block.bitmasked(ti.ijkl, (1,C1_BLOCK,H_BLOCK,W_BLOCK))
            pixel.place(self.x)

            self.active_uv = ti.field(ti.u1)
            self.active_block = ti.root.pointer(ti.ij, 
                                                (wrap_shape(h,H_BLOCK), wrap_shape(w,W_BLOCK)))
            active = self.active_block.bitmasked(ti.ij, (H_BLOCK,W_BLOCK))
            active.place(self.active_uv)
            self.init(indices, data, shape)
        else:
            self.x = x
            self.active_uv = active_uv

    @ti.kernel
    def init(self, indices:ti.types.template(), data:ti.types.template(), shape:ti.types.template()):
        b,c,h,w = shape
        n = indices.shape[0]
        for _b, _c, _n in ti.ndrange(b, c, n):
            _h, _w = indices[_n]
            self.x[_b, _c, _h, _w] = data[_n, _c]
        for _n in range(n):
            _h, _w = indices[_n]
            self.active_uv[_h, _w] = 1

    @ti.kernel
    def init_output(self,
                    kernel:ti.types.template(),
                    stride: ti.types.template(), padding: ti.types.template()):
        _, _, n_H_prev, n_W_prev = self.shape
        _, _, k_h, k_w = kernel.shape
        top_left_area = k_h*k_w
        s_h, s_w = stride
        p_h, p_w = padding
        max_h = n_H_prev - k_h + p_h
        max_w = n_W_prev - k_w + p_w
        if s_h > 1 or s_w > 1:
            for base_h, base_w in self.active_output_uv:
                for _k_h, _k_w in ti.ndrange(k_h, k_w):
                    _h = base_h - _k_h + p_h
                    _w = base_w - _k_w + p_w
                    if (_h >= 0 and _h % s_h == 0 and _h < max_h) and \
                        (_w >= 0 and _w % s_w == 0 and _w < max_w) and \
                            not ti.is_active(active_output_uv, _h // s_h, _w // s_w):
                        active_output_uv[_h // s_h, _w // s_w] = 1
        else:
            for base_h, base_w in self.active_output_uv:
                for _k_h, _k_w in ti.ndrange(k_h, k_w):
                    _h = base_h - _k_h + p_h
                    _w = base_w - _k_w + p_w
                    if (_h >= 0 and _h < max_h) and \
                        (_w >= 0 and _w < max_w) and \
                            not ti.is_active(active_output_uv, _h, _w):
                        active_output_uv[_h, _w] = 1
                # _h = ti.min(ti.max(base_h - _k_h + p_h, 0), n_H_prev - k_h + p_h)
                # _w = ti.min(ti.max(base_w - _k_w + p_w, 0), n_W_prev - k_w + p_w)
                # if _h % s_h == 0 and _w % s_w == 0:
                #     active_output_uv[_h, _w] = 1

    def conv(self, kernel: torch.Tensor, stride: list, padding: list):
        b, _, n_H_prev, n_W_prev = self.shape
        n_C, _, fh, fw = kernel.shape
        n_H = int((n_H_prev + padding[0] * 2 - fh)/ stride[0]) + 1
        n_W = int((n_W_prev + padding[1] * 2 - fw)/ stride[1]) + 1
        self.output_shape = [n_H, n_W]
        self.out_x = ti.filed(ti.f32)
        block = ti.root.pointer(ti.ijkl, (b, wrap_shape(n_C,C1_BLOCK), wrap_shape(n_H,H_BLOCK), wrap_shape(n_W,W_BLOCK)))
        pixel = block.bitmasked(ti.ijkl, (1,C1_BLOCK,H_BLOCK,W_BLOCK))
        pixel.place(self.out_x)

        self.active_output_uv = ti.field(ti.u1)
        self.output_active_block = ti.root.pointer(ti.ij, 
                                       (wrap_shape(n_H,H_BLOCK), wrap_shape(n_W,W_BLOCK)))
        active = active_block.bitmasked(ti.ij, (H_BLOCK,W_BLOCK))
        active.place(self.active_output_uv)
        self.init_output(kernel, stride, padding)

        self.conv_kernel(kernel, stride, padding)

        self.out_x = None
        self.active_output_uv = None

    @ti.kernel
    def conv_kernel(self, kernel: ti.types.template(),
                           stride: ti.types.template(), padding: ti.types.template()):
        """Sparsely iterate through all pixels.

        Args:
            kernel (ti.types.template): _description_
            stride (ti.types.template): _description_
            padding (ti.types.template): _description_
        """
        c1, c2, k_h, k_w = kernel.shape
        s_h, s_w = stride
        p_h, p_w = padding
        b, _, h_prev, w_prev = self.shape
        n_H, n_W = self.output_shape
        
        wrap_n_H = (h_prev + H_BLOCK - 1) // H_BLOCK
        wrap_n_W = (w_prev + W_BLOCK - 1) // W_BLOCK
        wrap_c1 = (c1 + C1_BLOCK - 1) // C1_BLOCK
        ti.loop_config(block_dim=H_BLOCK*W_BLOCK*C1_BLOCK)
        for _b, _c2, _n_c1, _n_h, _n_w, __c1, __h, __w in ti.ndrange(
            b, c2, wrap_c1, wrap_n_H, wrap_n_W, C1_BLOCK, H_BLOCK, W_BLOCK
        ):
            _c1 = __c1 + _n_c1 * C1_BLOCK
            _h = _n_h * H_BLOCK + __h
            _w = _n_w * W_BLOCK + __w
            if _h < h_prev and _w < w_prev and _c1 < c1\
                and ti.is_active(self.active_block, _n_h, _n_w):
                # block is active
                # collect input in this block
                inp = ti.simt.block.SharedArray((C1_BLOCK, H_BLOCK+7, W_BLOCK+7), ti.f32)
                ti.simt.block.sync()
                if ti.is_active(self.x, _c1, _h, _w):
                    inp[__c1, __h, __w] = self.x[_c1, _h, _w]
                else:
                    ... # collect inp from interpolated cache
                if __h < k_h and __w < k_w:
                    e_h = _h + H_BLOCK
                    e_w = _w + W_BLOCK
                    if e_h < h_prev and e_w < w_prev:
                        if ti.is_active(self.x, _c1, _h, _w):
                            inp[__c1, e_h, e_w] = self.x[_c1, _h + H_BLOCK, _w + W_BLOCK]
                        else:
                            ... # collect inp from interpolated cache
                    else:
                        inp[__c1, e_h, e_w] = 0.
                ti.simt.block.sync()

    @ti.kernel
    def conv_kernel_stride_1(self, kernel: ti.types.template(),
                           stride: ti.types.template(), padding: ti.types.template()):
        """Sparsely iterate through all pixels.

        Args:
            kernel (ti.types.template): _description_
            stride (ti.types.template): _description_
            padding (ti.types.template): _description_
        """
        c1, c2, k_h, k_w = kernel.shape
        s_h, s_w = stride
        p_h, p_w = padding
        b, _, h_prev, w_prev = self.shape
        n_H, n_W = self.output_shape
        
        wrap_n_H = (n_H + H_BLOCK - 1) // H_BLOCK
        wrap_n_W = (n_W + W_BLOCK - 1) // W_BLOCK
        wrap_c1 = (c1 + C1_BLOCK - 1) // C1_BLOCK
        wrap_c2 = (c2 + C2_BLOCK - 1) // C2_BLOCK
        ti.loop_config(block_dim=H_BLOCK*W_BLOCK*C1_BLOCK*C2_BLOCK)
        for _b, _n_c2, _n_c1, _n_h, _n_w, __c2, __c1, __h, __w in ti.ndrange(
            b, wrap_c2, wrap_c1, wrap_n_H, wrap_n_W, C2_BLOCK, C1_BLOCK, H_BLOCK, W_BLOCK
        ):
            _c1 = __c1 + _n_c1 * C1_BLOCK
            _c2 = __c2 + _n_c2 * C2_BLOCK
            _h = _n_h * H_BLOCK + __h
            _w = _n_w * W_BLOCK + __w
            _h_inp = _h - p_h
            _w_inp = _w - p_w
            if _c1 < c1 and _c2 < c2 \
                and ti.is_active(self.output_active_block, _n_h, _n_w):
                # block is active
                # collect input in this block
                inp = ti.simt.block.SharedArray((C1_BLOCK, H_BLOCK+7, W_BLOCK+7), ti.f32)
                out = ti.simt.block.SharedArray((C2_BLOCK, H_BLOCK, W_BLOCK), ti.f32)
                ti.simt.block.sync()
                if __c1 == 0:
                    out[__c2, __h, __w] = 0.
                if __c2 == 0:
                    if _h < n_H and _w < n_W and ti.is_active(self.x, _c1, _h_inp, _w_inp) :
                        inp[__c1, __h, __w] = self.x[_c1, _h_inp, _w_inp]
                    else:
                        ... # collect inp from interpolated cache
                    if __h < k_h and __w < k_w:
                        e_h = _h_inp + H_BLOCK
                        e_w = _w_inp + W_BLOCK
                        if e_h < h_prev: 
                            if ti.is_active(self.x, 0, e_h, _w_inp):
                                inp[__c1, __h + H_BLOCK, __w] = self.x[_c1, e_h, _w_inp]
                            else:
                                ...
                        else:
                            inp[__c1, __h + H_BLOCK, __w] = 0.
                        if e_w < w_prev:
                            if ti.is_active(self.x, 0, _h_inp, e_w):
                                inp[__c1, __h, __w+W_BLOCK] = self.x[_c1, _h_inp, e_w]
                            else:
                                ...
                        else:
                            inp[__c1, __h, __w+W_BLOCK] = 0.
                        if e_h < h_prev and e_w < w_prev:
                            if ti.is_active(self.x, 0, e_h, e_w):
                                inp[__c1, __h+H_BLOCK, __w+W_BLOCK] = self.x[
                                    _c1, e_h, e_w]
                            else:
                                ...
                        else:
                            inp[__c1, __h+H_BLOCK, __w+W_BLOCK] = 0.
                ti.simt.block.sync()
                if _h < n_H and _w < n_W and \
                    ti.is_active(self.out_x, _h, _w):
                    _out = 0.
                    for _k_h, _k_w in ti.ndrange(k_h, k_w):
                        _out += inp[__c1, __h + _k_h, __w + _k_w] * kernel[
                            c1, c2, _k_h, _k_w]
                    out[__c2, __h, __w] += _out
                    ti.simt.block.sync()
                    self.out_x[0, c2, _h, _w] = out[__c2, __h, __w]
