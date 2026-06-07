"""模型构建工厂：统一管理 timm 原生模型与所有历史沉淀的本地非标模型。"""

from __future__ import annotations

from pathlib import Path

import timm
from torch import nn

from utils.config import ModelConfig
from utils.checkpoint import load_weights_flexible

# --------------------------------------------------------
# [本地非标模型注册区]
# 容错引入：即使缺失某文件夹，也不影响其他标准模型的运行
# --------------------------------------------------------
try:
    from models.cswin.cswin import CSWin_96_24322_base_384
except ImportError:
    CSWin_96_24322_base_384 = None


def build_model(
    config: ModelConfig,
    pretrained_path: Path | None = None,
    require_pretrained: bool = False,
) -> nn.Module:
    """创建单节点二分类模型工厂。"""

    # ==========================================
    # 1. 历史遗产 / 本地非标模型路由
    # ==========================================
    if config.name == "cswin_base_384":
        if CSWin_96_24322_base_384 is None:
            raise RuntimeError("缺少 CSWin 源码，无法复现该模型。请检查 models/cswin/ 目录。")
        model = CSWin_96_24322_base_384(
            pretrained=False, 
            num_classes=1, 
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            img_size=384  # 强制纠正分辨率崩塌的历史问题
        )

    # 如果未来有其他非标模型，继续写 elif config.name == "custom_model_v2":

    # ==========================================
    # 2. timm 标准模型兜底 (当前主力：BEiT 等)
    # ==========================================
    else:
        model = timm.create_model(
            config.name,
            pretrained=False,
            num_classes=1,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
        )

    # ==========================================
    # 3. 统一的权重加载与维度兼容逻辑
    # ==========================================
    if pretrained_path is not None and pretrained_path.is_file():
        loaded_count, skipped = load_weights_flexible(model, pretrained_path)
        print(
            f"已载入预训练参数 {loaded_count} 项: {pretrained_path}; "
            f"跳过 {len(skipped)} 项不兼容参数。"
        )
    elif require_pretrained:
        raise FileNotFoundError(f"缺少预训练参数文件: {pretrained_path}，请先放入 weights/。")
        
    return model