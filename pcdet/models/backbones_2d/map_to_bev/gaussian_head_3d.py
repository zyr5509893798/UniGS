import torch
import torch.nn as nn


class GaussianHead3d(nn.Module):
    """
    3D Gaussian Head for UniTR model
    Converts voxel features into Gaussian parameters for 3D Gaussian Splatting

    Args:
        input_dim: Input feature dimension (from UniTR)
        gaussian_dim: Output dimension for Gaussian parameters
        sh_degree: Degree of spherical harmonics (default=1)
        use_offsets: Whether to predict offsets from voxel centers
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of MLP layers
    """

    def __init__(self, input_dim, gaussian_dim=20, sh_degree=1,
                 use_offsets=True, hidden_dim=256, num_layers=3):
        super().__init__()
        # 球谐函数阶数（控制颜色表示复杂度）
        self.sh_degree = sh_degree
        # 是否预测体素中心偏移
        self.use_offsets = use_offsets

        # 计算高斯参数各分量的维度
        self.offset_dim = 3 if use_offsets else 0  # 偏移量维度（x,y,z）
        self.scale_dim = 3  # 缩放因子维度（x,y,z）
        self.rotation_dim = 4  # 旋转四元数维度（w,x,y,z）
        self.sh_dim = 3 * sh_degree  # 球谐系数维度（RGB各通道）
        self.opacity_dim = 1  # 不透明度维度

        # 构建MLP网络
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = gaussian_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:  # 除最后一层外都添加ReLU激活
                layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*layers)  # 序列化MLP

        # 初始化权重
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier均匀初始化
                nn.init.constant_(layer.bias, 0)  # 偏置初始化为0

    def forward(self, batch_dict):
        """
        Args:
            batch_dict (dict):
                - 'pillar_features': Voxel features [N, C]
                - 'voxel_coords': Voxel coordinates [N, 4] (batch_id, z, y, x)

        Returns:
            batch_dict (dict) with added Gaussian parameters:
                - 'gaussian_means': [N, 3] (x, y, z)
                - 'gaussian_scales': [N, 3]
                - 'gaussian_rotations': [N, 4] (quaternions)
                - 'gaussian_sh': [N, 3*sh_degree] (spherical harmonics)
                - 'gaussian_opacities': [N, 1]
        """
        # 提取输入特征和坐标
        features = batch_dict['pillar_features']  # [N, C] 体素特征
        coords = batch_dict['voxel_coords']  # [N, 4] (batch_id, z, y, x)

        # 转换坐标格式：从(z,y,x)到(x,y,z)
        x = coords[:, 3]  # x坐标（第3列）
        y = coords[:, 2]  # y坐标（第2列）
        z = coords[:, 1]  # z坐标（第1列）
        voxel_positions = torch.stack([x, y, z], dim=1)  # [N, 3] (x,y,z)

        # 通过MLP预测高斯参数
        gaussian_params = self.mlp(features)  # [N, gaussian_dim]

        # 分割高斯参数为各分量
        start_idx = 0

        # 1. 偏移量（可选）
        if self.use_offsets:
            offsets = gaussian_params[:, :3]  # 取前3个值作为偏移
            start_idx += 3
        else:
            offsets = torch.zeros_like(voxel_positions)  # 无偏移则设为零

        # 2. 缩放因子（取接下来的3个值）
        scales = gaussian_params[:, start_idx:start_idx + self.scale_dim]
        start_idx += self.scale_dim

        # 3. 旋转四元数（取接下来的4个值）
        rotations = gaussian_params[:, start_idx:start_idx + self.rotation_dim]
        start_idx += self.rotation_dim

        # 4. 球谐系数（取接下来的sh_dim个值）
        sh = gaussian_params[:, start_idx:start_idx + self.sh_dim]
        start_idx += self.sh_dim

        # 5. 不透明度（取最后1个值）
        opacities = gaussian_params[:, start_idx:start_idx + self.opacity_dim]

        # 应用激活函数约束参数范围
        scales = reg_dense_scales(scales)  # 指数激活确保正数
        rotations = reg_dense_rotation(rotations)  # 归一化为单位四元数
        opacities = reg_dense_opacities(opacities)  # Sigmoid激活到[0,1]

        # 计算最终位置：体素中心 + 偏移量
        means = voxel_positions + offsets

        # 存储结果到batch_dict
        batch_dict['gaussian_means'] = means
        batch_dict['gaussian_scales'] = scales
        batch_dict['gaussian_rotations'] = rotations
        batch_dict['gaussian_sh'] = sh
        batch_dict['gaussian_opacities'] = opacities

        return batch_dict


# 偏移量激活函数（无操作）
def reg_dense_offsets(offsets):
    return offsets

# 缩放因子激活函数（指数确保正值）
def reg_dense_scales(scales):
    return torch.exp(scales)

# 旋转四元数归一化
def reg_dense_rotation(rotations, eps=1e-8):
    return rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

# 球谐系数激活函数（无操作）
def reg_dense_sh(sh):
    return sh

# 不透明度激活函数（Sigmoid到[0,1]）
def reg_dense_opacities(opacities):
    return torch.sigmoid(opacities)