import torch.nn as nn
from mmcv.runner import auto_fp16
import torch
import warnings
from depth.models.builder import HEADS
from depth.models.decode_heads.decode_head import DepthBaseDecodeHead

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmcv.cnn.bricks.transformer import (build_transformer_layer_sequence,
                                         build_positional_encoding)
from mmcv.cnn import ConvModule


@HEADS.register_module()
class DeformableHeadWithTime(DepthBaseDecodeHead):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """
    
    def __init__(self,
                 num_feature_levels,
                 encoder,
                 positional_encoding,
                 **kwargs):
        
        super().__init__(input_transform='multiple_select', **kwargs)
    
        self.num_feature_levels = num_feature_levels
        self.encoder = build_transformer_layer_sequence(encoder)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
                                                 f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
                                                 f' and {num_feats}.'
        # self.level_embeds = nn.Parameter(
        #     torch.Tensor(self.num_feature_levels, self.embed_dims))
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        # normal_(self.level_embeds)
    
    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Get the reference points used in decoder.
        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points
    
    @auto_fp16()
    def forward(self, inputs, times):
        
        mlvl_feats = inputs[-self.num_feature_levels:]
        
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        out = self.depth_pred(memory)
        return out
    
    def forward_train(self, inputs, times, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, times, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, times)


@HEADS.register_module()
class SpadeDeformableHeadWithTime(DeformableHeadWithTime):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @auto_fp16()
    def forward(self, inputs, times, condition):
        mlvl_feats = inputs[-self.num_feature_levels:]
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            condition=condition,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        out = self.conv_seg(memory)
        return out

    def forward_train(self, inputs, times, condition, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times, condition)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, times, condition, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, times, condition)
    

class PixelShuffle(nn.Module):

    def __init__(self, ):
        super(PixelShuffle, self).__init__()

    def forward(self, x, scale_factor):
        n, c, h, w = x.size()
        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)

        return x
    
    
@HEADS.register_module()
class SpadeDeformableHeadWithTimeUpConv(DeformableHeadWithTime):
    """Implements the DeformableEncoder.
    Args:
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ps = PixelShuffle()
        self.conv = ConvModule(
            in_channels=self.embed_dims // 4,
            out_channels=self.embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            act_cfg=dict(type='ReLU'))
        self.conv_seg = nn.Conv2d(self.embed_dims // 16, self.out_channels,
                                  kernel_size=3, padding=1)

    @auto_fp16()
    def forward(self, inputs, times, condition):
        mlvl_feats = inputs[-self.num_feature_levels:]
        feat_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = torch.zeros((bs, h, w), device=feat.device, requires_grad=False)
            pos_embed = self.positional_encoding(mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            feat = feat.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        reference_points = self.get_reference_points(spatial_shapes, device=feat.device)
        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            condition=condition,
            key=None,
            value=None,
            time=times,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=None,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index)
        memory = memory.permute(1, 2, 0)
        memory = memory.reshape(bs, c, h, w).contiguous()
        memory = self.ps(memory, scale_factor=2.0)
        memory = self.conv(memory)
        memory = self.ps(memory, scale_factor=2.0)
        out = self.conv_seg(memory)
        return out

    def forward_train(self, inputs, times, condition, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self(inputs, times, condition)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, times, condition, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, times, condition)