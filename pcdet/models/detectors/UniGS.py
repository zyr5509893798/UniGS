import torch
import torch.nn as nn
from ...utils.spconv_utils import find_all_spconv_keys
from .detector3d_template import Detector3DTemplate
from .. import mm_backbone
from ..pixelsplat.decoder_splatting_cuda import DecoderSplattingCUDA
from ..backbones_2d.map_to_bev.gaussian_head_3d import GaussianHead3d


class UniGS(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        # 简化的模块拓扑结构，只保留3D特征提取和高斯重建部分
        self.module_topology = [
            'vfe', 'mm_backbone', 'gaussian_head'
        ]

        self.module_list = self.build_networks()

        # 初始化渲染器（不作为网络模块，仅在需要时使用）
        background_color = self.model_cfg.get('BACKGROUND_COLOR', [0.0, 0.0, 0.0])
        self.renderer = DecoderSplattingCUDA(background_color=background_color)

    def build_mm_backbone(self, model_info_dict):
        if self.model_cfg.get('MM_BACKBONE', None) is None:
            return None, model_info_dict

        mm_backbone_name = self.model_cfg.MM_BACKBONE.NAME
        mm_backbone_module = mm_backbone.__all__[mm_backbone_name](
            model_cfg=self.model_cfg.MM_BACKBONE
        )  # 这个位置关联到了真正的Uni模型，将模型参数传递到了真正的 Uni模型。
        model_info_dict['module_list'].append(mm_backbone_module)

        # 记录输出特征维度
        if hasattr(mm_backbone_module, 'get_output_feature_dim'):
            model_info_dict['num_point_features'] = mm_backbone_module.get_output_feature_dim()

        return mm_backbone_module, model_info_dict

    def build_gaussian_head(self, model_info_dict):
        if self.model_cfg.get('GAUSSIAN_HEAD', None) is None:
            return None, model_info_dict

        # 获取输入特征维度
        input_dim = model_info_dict.get('num_point_features', 256)

        gaussian_head_module = GaussianHead3d(
            input_dim=input_dim,
            gaussian_dim=self.model_cfg.GAUSSIAN_HEAD.get('GAUSSIAN_DIM', 20),
            sh_degree=self.model_cfg.GAUSSIAN_HEAD.get('SH_DEGREE', 1),
            use_offsets=self.model_cfg.GAUSSIAN_HEAD.get('USE_OFFSETS', True),
            hidden_dim=self.model_cfg.GAUSSIAN_HEAD.get('HIDDEN_DIM', 256),
            num_layers=self.model_cfg.GAUSSIAN_HEAD.get('NUM_LAYERS', 3)
        )
        model_info_dict['module_list'].append(gaussian_head_module)
        return gaussian_head_module, model_info_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # 处理spconv权重形状不匹配的问题
                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            # 适配预训练权重命名
            if 'image_backbone' in key:
                key = key.replace("image", "mm")
                if 'input_layer' in key:
                    key = key.replace("input_layer", "image_input_layer")

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            else:
                print("权重不匹配:", key)

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def forward(self, batch_dict):
        # 前向传播通过所有模块
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # 如果提供了相机参数，则进行渲染
        if 'camera_params' in batch_dict and not self.training:
            # 提取高斯参数
            gaussian_params = {
                'means': batch_dict['gaussian_means'],
                'scales': batch_dict['gaussian_scales'],
                'rotations': batch_dict['gaussian_rotations'],
                'sh': batch_dict['gaussian_sh'],
                'opacities': batch_dict['gaussian_opacities']
            }

            # 使用渲染器生成图像
            rendered_images = self.renderer(
                batch_dict,
                gaussian_params,
                batch_dict['camera_params']
            )

            batch_dict['rendered_images'] = rendered_images

            return rendered_images

        return batch_dict

    def get_training_loss(self, batch_dict):
        # 计算重建损失
        # 这里需要根据您的实际损失函数实现
        # 例如，可以比较渲染图像与真实图像的差异

        # 首先进行渲染
        gaussian_params = {
            'means': batch_dict['gaussian_means'],
            'scales': batch_dict['gaussian_scales'],
            'rotations': batch_dict['gaussian_rotations'],
            'sh': batch_dict['gaussian_sh'],
            'opacities': batch_dict['gaussian_opacities']
        }

        rendered_images = self.renderer(
            batch_dict,
            gaussian_params,
            batch_dict['camera_params']
        )

        # 计算重建损失
        reconstruction_loss = torch.nn.functional.mse_loss(
            rendered_images,
            batch_dict['target_images']
        )

        tb_dict = {
            'reconstruction_loss': reconstruction_loss.item(),
        }

        loss = reconstruction_loss
        return loss, tb_dict, {}