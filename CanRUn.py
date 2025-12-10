# from ultralytics import YOLO
# from ultralytics.nn.modules import ConcatWithDDAWA
#
# # 检查模块是否可导入
# print("ConcatWithDDAWA available:", hasattr(ConcatWithDDAWA, 'forward'))
#
# # 尝试构建模型
# model = YOLO('yolov8s_ddawa.yaml')
# print("Model created successfully!")
#
# # 打印模型结构，查看是否包含自定义模块
# print(model.model)

# 示例代码：验证前向传播
import torch
from ultralytics import YOLO

model = YOLO('yolov8s_ddawa.yaml')
dummy_input = torch.rand(1, 3, 640, 640)
output = model(dummy_input)

# 检查输出是否包含SHaSM模块的特征
print(output[0].shape)  # 应输出检测结果的形状