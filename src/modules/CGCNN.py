from __future__ import print_function, division

import torch
import torch.nn as nn

# adapted from https://github.com/txie-93/cgcnn/blob/master/cgcnn/model.py
class ConvLayer(nn.Module):
    """
    晶体图卷积层 - 在图上进行卷积操作，聚合邻居原子信息
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        初始化卷积层

        参数
        ----------
        atom_fea_len: int
          原子隐藏特征的维度
        nbr_fea_len: int
          键特征的维度
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        
        # 核心全连接层：将中心原子+邻居原子+键特征映射到门控空间
        # 输入维度: 中心原子(at) + 邻居原子(nbr) + 键特征 = atom_fea_len*2 + nbr_fea_len
        # 输出维度: 2*atom_fea_len (分为门控信号和核心信息两部分)
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()      # 用于门控信号，输出0-1的权重
        self.softplus1 = nn.Softplus()   # 用于核心信息，保证正值
        self.softplus2 = nn.Softplus()   # 用于最终输出
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)  # 用于门控和核心信息的BN
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)    # 用于聚合结果的BN

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        前向传播过程

        参数说明
        ----------
        atom_in_fea: torch.Tensor, shape (N, atom_fea_len)
          卷积前的原子隐藏特征，N是批次中的总原子数
        nbr_fea: torch.Tensor, shape (N, M, nbr_fea_len)  
          每个原子的M个邻居的键特征
        nbr_fea_idx: torch.LongTensor, shape (N, M)
          每个原子的M个邻居的索引

        返回
        -------
        atom_out_fea: torch.Tensor, shape (N, atom_fea_len)
          卷积后的原子隐藏特征
        """
        # 获取张量形状
        N, M = nbr_fea_idx.shape  # N=总原子数, M=最大邻居数
        
        # === 步骤1: 收集邻居原子特征 ===
        # 根据邻居索引获取所有邻居原子的特征
        # atom_nbr_fea shape: (N, M, atom_fea_len)
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        
        # === 步骤2: 构建局部环境特征 ===
        # 将中心原子特征扩展为与邻居相同的维度
        # atom_in_fea.unsqueeze(1): (N, 1, atom_fea_len) -> .expand(N, M, atom_fea_len): (N, M, atom_fea_len)
        center_atom_expanded = atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)
        
        # 拼接特征: [中心原子, 邻居原子, 键特征]
        # total_nbr_fea shape: (N, M, 2*atom_fea_len + nbr_fea_len)
        total_nbr_fea = torch.cat([center_atom_expanded, atom_nbr_fea, nbr_fea], dim=2)
        
        # === 步骤3: 通过全连接层生成门控和核心信息 ===
        total_gated_fea = self.fc_full(total_nbr_fea)  # shape: (N, M, 2*atom_fea_len)
        
        # 应用批归一化 - 需要调整形状以适应BN层
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        
        # === 步骤4: 分割出门控信号和核心信息 ===
        # 将输出分割为两部分: 门控滤波器(nbr_filter)和核心信息(nbr_core)
        # 每部分shape: (N, M, atom_fea_len)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        
        # 应用激活函数
        nbr_filter = self.sigmoid(nbr_filter)  # 门控信号，范围[0,1]，决定信息传递权重
        nbr_core = self.softplus1(nbr_core)    # 核心信息，使用Softplus保证正值
        
        # === 步骤5: 信息聚合 ===
        # 对邻居信息进行加权求和: nbr_filter * nbr_core 然后沿邻居维度求和
        # nbr_sumed shape: (N, atom_fea_len)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        
        # 应用批归一化
        nbr_sumed = self.bn2(nbr_sumed)
        
        # === 步骤6: 更新原子特征 (残差连接) ===
        # 使用残差连接: 新特征 = 原特征 + 聚合的邻居信息
        out = self.softplus2(atom_in_fea + nbr_sumed)
        
        return out


class CrystalGraphConvNet(nn.Module):
    """
    晶体图卷积神经网络 - 用于预测材料整体性质
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        初始化晶体图卷积网络

        参数
        ----------
        orig_atom_fea_len: int
          输入原子特征的原始维度
        nbr_fea_len: int
          键特征的维度
        atom_fea_len: int
          卷积层中原子隐藏特征的维度
        n_conv: int
          卷积层的数量
        h_fea_len: int
          池化后的隐藏特征维度
        n_h: int
          池化后的全连接层数量
        classification: bool
          是否为分类任务（False为回归任务）
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        
        # === 特征嵌入层 ===
        # 将原始原子特征映射到高维隐藏空间
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        
        # === 多层图卷积 ===
        # 创建多个卷积层，每层都会聚合更远距离的邻居信息
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        
        # === 池化后的全连接层 ===
        # 将原子级特征转换为晶体级特征后的处理层
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # === 可选的多层全连接 ===
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        
        # === 输出层 ===
        # 分类任务输出2个类别，回归任务输出1个值
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
            
        # === 分类任务专用层 ===
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)  # 输出对数概率
            self.dropout = nn.Dropout()             # 防止过拟合

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        前向传播

        参数说明
        ----------
        atom_fea: torch.Tensor, shape (N, orig_atom_fea_len)
          基于原子类型的原子特征
        nbr_fea: torch.Tensor, shape (N, M, nbr_fea_len)
          每个原子的M个邻居的键特征
        nbr_fea_idx: torch.LongTensor, shape (N, M)
          每个原子的M个邻居的索引
        crystal_atom_idx: list of torch.LongTensor, length N0
          从晶体索引到原子索引的映射，N0是批次中的晶体总数

        返回
        -------
        prediction: torch.Tensor
          模型的预测输出
        """
        # === 步骤1: 特征嵌入 ===
        # 将原始原子特征映射到高维空间
        atom_fea = self.embedding(atom_fea)  # shape: (N, atom_fea_len)
        
        # === 步骤2: 多层图卷积 ===
        # 依次通过多个卷积层，每层聚合邻居信息并更新原子特征
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
            # 每经过一层卷积，原子的感受野增大，能捕获更远距离的化学环境信息
        
        # === 步骤3: 池化 - 原子特征 -> 晶体特征 ===
        # 将每个晶体的所有原子特征聚合成一个晶体级特征向量
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)  # shape: (N0, atom_fea_len)
        
        # === 步骤4: 全连接层处理 ===
        # 第一个全连接层 + 激活函数
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # 分类任务使用dropout防止过拟合
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        
        # === 步骤5: 可选的多层全连接 ===
        # 如果有额外的全连接层，依次通过
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        # === 步骤6: 最终输出 ===
        out = self.fc_out(crys_fea)  # 最终预测
        
        # 分类任务应用log softmax
        if self.classification:
            out = self.logsoftmax(out)
            
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        池化操作 - 将原子特征聚合成晶体特征

        参数
        ----------
        atom_fea: torch.Tensor, shape (N, atom_fea_len)
          批次的原子特征向量
        crystal_atom_idx: list of torch.LongTensor, length N0
          从晶体索引到原子索引的映射

        返回
        -------
        pooled_fea: torch.Tensor, shape (N0, atom_fea_len)
          池化后的晶体特征
        """
        # 验证原子索引的完整性
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == atom_fea.data.shape[0]
        
        # 对每个晶体的所有原子特征取平均值
        # 这里使用平均池化，也可以尝试最大池化或求和池化
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        
        # 将所有晶体的特征拼接成张量
        return torch.cat(summed_fea, dim=0)