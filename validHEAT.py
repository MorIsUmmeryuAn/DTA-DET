import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings

warnings.filterwarnings('ignore')


class YOLOv8DDAWAHeatmapVisualizer:
    def __init__(self, model_path, img_size=640):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.feature_maps = {}
        self.attention_maps = {}
        self.hooks = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        try:
            # 加载检查点文件
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 检查是否是Ultralytics格式的检查点
            if 'model' in checkpoint:
                # 这是Ultralytics格式的检查点
                self.model = checkpoint['model']
                print(f"✅ 成功加载模型 (检查点格式)")
                print(f"训练轮次: {checkpoint.get('epoch', 'N/A')}")
                print(f"最佳fitness: {checkpoint.get('best_fitness', 'N/A')}")
                print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

                # 确保模型在正确的设备上
                self.model = self.model.to(self.device)

                # 统一数据类型
                if next(self.model.parameters()).dtype == torch.float16:
                    print("⚠️ 检测到半精度模型，转换为单精度")
                    self.model = self.model.float()

                return True
            elif hasattr(checkpoint, 'model'):
                # 直接模型对象
                self.model = checkpoint.model
                print(f"✅ 成功加载YOLOv8-DDAWA模型 (直接模型)")
                self.model = self.model.to(self.device).float()
                return True
            else:
                # 尝试作为权重文件处理
                print("⚠️ 尝试作为权重文件处理")
                self.model = checkpoint
                self.model = self.model.to(self.device).float()
                print(f"✅ 成功加载模型权重")
                return True

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            return False

    def register_hooks(self):
        """注册钩子来捕获特征图和注意力图"""
        if self.model is None:
            return False

        # 查找所有ConcatWithDDAWA模块
        ddawa_count = 0
        for name, module in self.model.named_modules():
            if 'ConcatWithDDAWA' in str(type(module)):
                # 为每个DDAWA模块注册钩子
                hook = module.register_forward_hook(self._get_ddawa_hook(name))
                self.hooks.append(hook)
                ddawa_count += 1
                print(f"✅ 注册钩子到DDAWA模块: {name}")

        # 查找Detect层
        detect_count = 0
        for name, module in self.model.named_modules():
            if 'Detect' in str(type(module)):
                hook = module.register_forward_hook(self._get_detect_hook(name))
                self.hooks.append(hook)
                detect_count += 1
                print(f"✅ 注册钩子到Detect层: {name}")

        # 查找关键卷积层
        conv_count = 0
        target_layers = ['10', '14', '18', '21']  # 关键卷积层
        for name, module in self.model.named_modules():
            if any(target in name for target in target_layers) and isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(self._get_conv_hook(name))
                self.hooks.append(hook)
                conv_count += 1
                print(f"✅ 注册钩子到关键层: {name}")

        print(f"✅ 总共注册了 {ddawa_count} 个DDAWA模块, {detect_count} 个Detect层, {conv_count} 个关键层")
        return True

    def _get_ddawa_hook(self, name):
        """创建DDAWA模块的钩子函数"""

        def hook(module, input, output):
            try:
                # 捕获注意力权重
                attention_data = {}

                # 检查是否有ddawa_modules属性
                if hasattr(module, 'ddawa_modules'):
                    for i, ddawa_module in enumerate(module.ddawa_modules):
                        # 捕获通道注意力
                        if hasattr(ddawa_module, 'channel_attention'):
                            channel_att = ddawa_module.channel_attention
                            # 尝试获取最终的注意力权重
                            if hasattr(channel_att, 'weight'):
                                attention_data[f'channel_att_{i}'] = channel_att.weight.detach().cpu().numpy()

                        # 捕获空间注意力
                        if hasattr(ddawa_module, 'spatial_attention'):
                            spatial_att = ddawa_module.spatial_attention
                            # 这里需要根据具体实现获取权重
                            attention_data[f'spatial_att_{i}'] = None

                if attention_data:
                    self.attention_maps[name] = attention_data

                # 捕获输出特征图
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu().numpy()

            except Exception as e:
                print(f"❌ DDAWA钩子错误 ({name}): {e}")

        return hook

    def _get_detect_hook(self, name):
        """创建Detect层的钩子函数"""

        def hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu().numpy()
                # 捕获检测结果
                self.detection_output = output
            except Exception as e:
                print(f"❌ Detect钩子错误: {e}")

        return hook

    def _get_conv_hook(self, name):
        """创建卷积层的钩子函数"""

        def hook(module, input, output):
            try:
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu().numpy()
            except Exception as e:
                print(f"❌ Conv钩子错误 ({name}): {e}")

        return hook

    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img_rgb.copy()

        # 保存原始尺寸
        h, w = img_rgb.shape[:2]

        # 调整大小（保持宽高比）
        scale = min(self.img_size / w, self.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # 调整图像大小
        resized_img = cv2.resize(img_rgb, (new_w, new_h))

        # 创建填充图像
        padded_img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # 转换为模型输入格式（确保数据类型匹配）
        input_tensor = torch.from_numpy(padded_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(self.device)

        # 确保输入数据类型与模型一致
        if next(self.model.parameters()).dtype == torch.float32:
            input_tensor = input_tensor.float()
        elif next(self.model.parameters()).dtype == torch.float16:
            input_tensor = input_tensor.half()

        return input_tensor, original_img, (w, h), scale

    def generate_heatmaps(self, image_path, output_dir="yolov8_ddawa_heatmaps"):
        """生成热力图"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载模型
        if not self.load_model():
            print("❌ 模型加载失败")
            return

        # 设置为评估模式
        self.model.eval()

        # 注册钩子
        if not self.register_hooks():
            print("❌ 钩子注册失败")
            return

        # 预处理图像
        try:
            input_tensor, original_img, original_size, scale = self.preprocess_image(image_path)
            print(f"✅ 图像预处理完成: 原始尺寸 {original_size}, 缩放比例 {scale:.3f}")
            print(f"📊 输入张量形状: {input_tensor.shape}, 数据类型: {input_tensor.dtype}")
            print(f"📊 模型参数数据类型: {next(self.model.parameters()).dtype}")
        except Exception as e:
            print(f"❌ 图像预处理失败: {e}")
            return

        # 前向传播
        print("🚀 进行前向传播...")
        try:
            with torch.no_grad():
                # 确保输入和模型在同一设备上
                input_tensor = input_tensor.to(self.device)
                if next(self.model.parameters()).dtype == torch.float16:
                    input_tensor = input_tensor.half()
                else:
                    input_tensor = input_tensor.float()

                output = self.model(input_tensor)
            print("✅ 前向传播完成")
        except Exception as e:
            print(f"❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 移除钩子
        for hook in self.hooks:
            hook.remove()

        print(f"📊 捕获到 {len(self.feature_maps)} 个特征图")
        print(f"📊 捕获到 {len(self.attention_maps)} 个注意力图")

        # 保存原始图像
        plt.figure(figsize=(10, 8))
        plt.imshow(original_img)
        plt.title('Original Image', fontsize=16)
        plt.axis('off')
        plt.savefig(f'{output_dir}/original_image.jpg', dpi=300, bbox_inches='tight')
        plt.close()

        # 可视化特征热力图
        self._visualize_feature_maps(original_img, output_dir)

        # 可视化注意力热力图
        if self.attention_maps:
            self._visualize_attention_maps(original_img, output_dir)

        print(f"✅ 热力图生成完成！结果保存在: {output_dir}/")

    def _visualize_feature_maps(self, original_img, output_dir):
        """可视化特征图"""
        if not self.feature_maps:
            print("⚠️ 没有捕获到特征图")
            return

        # 选择重要的层进行可视化
        important_layers = []
        for name in self.feature_maps.keys():
            # 选择包含DDAWA、Detect或关键数字的层
            if 'ConcatWithDDAWA' in name or 'Detect' in name or any(str(i) in name for i in [10, 12, 14, 16, 18, 21]):
                important_layers.append(name)

        if not important_layers:
            important_layers = list(self.feature_maps.keys())[:8]  # 取前8个

        # 创建特征图可视化
        n_cols = min(4, len(important_layers))
        n_rows = (len(important_layers) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, layer_name in enumerate(important_layers):
            row = i // n_cols
            col = i % n_cols

            if row >= n_rows:
                break

            feature_map = self.feature_maps[layer_name]

            try:
                # 处理不同形状的特征图
                if len(feature_map.shape) == 4:  # [batch, channels, height, width]
                    # 取第一个batch和所有通道的均值
                    heatmap = np.mean(feature_map[0], axis=0)
                elif len(feature_map.shape) == 3:  # [channels, height, width]
                    heatmap = np.mean(feature_map, axis=0)
                elif len(feature_map.shape) == 2:  # [height, width]
                    heatmap = feature_map
                else:
                    continue

                # 调整到原始图像尺寸
                h, w = original_img.shape[:2]
                heatmap_resized = cv2.resize(heatmap, (w, h))

                # 归一化
                heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (
                            heatmap_resized.max() - heatmap_resized.min() + 1e-8)

                # 显示热力图
                axes[row, col].imshow(heatmap_resized, cmap='jet', alpha=0.7)
                axes[row, col].imshow(original_img, alpha=0.5)
                axes[row, col].set_title(f'{layer_name}\n{feature_map.shape}', fontsize=9)
                axes[row, col].axis('off')

            except Exception as e:
                print(f"❌ 处理特征图 {layer_name} 失败: {e}")
                axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(len(important_layers), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_heatmaps.jpg', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 特征热力图已保存")

    def _visualize_attention_maps(self, original_img, output_dir):
        """可视化注意力图"""
        if not self.attention_maps:
            print("⚠️ 没有捕获到注意力图")
            return

        for layer_name, attention_data in self.attention_maps.items():
            print(f"📋 处理注意力层: {layer_name}")

            # 创建注意力可视化
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 通道注意力可视化
            if 'channel_att_0' in attention_data and attention_data['channel_att_0'] is not None:
                channel_att = attention_data['channel_att_0']
                axes[0].bar(range(len(channel_att)), channel_att)
                axes[0].set_title(f'{layer_name} - Channel Attention')
                axes[0].set_xlabel('Channel Index')
                axes[0].set_ylabel('Attention Weight')
                axes[0].grid(True, alpha=0.3)
            else:
                axes[0].text(0.5, 0.5, 'Channel Attention Data\nNot Available',
                             ha='center', va='center', fontsize=12)
                axes[0].set_title(f'{layer_name} - Channel Attention')
                axes[0].axis('off')

            # 空间注意力占位符
            axes[1].text(0.5, 0.5, 'Spatial Attention\n(需要具体实现)',
                         ha='center', va='center', fontsize=12)
            axes[1].set_title(f'{layer_name} - Spatial Attention')
            axes[1].axis('off')

            plt.tight_layout()
            safe_name = layer_name.replace('.', '_').replace('(', '').replace(')', '')
            plt.savefig(f'{output_dir}/attention_{safe_name}.jpg', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ 注意力图已保存")


# 使用示例
if __name__ == "__main__":
    print("🚀 开始YOLOv8-DDAWA热力图分析...")

    # 配置参数
    model_path = ".pt"  # 您的YOLOv8-DDAWA模型文件
    image_path = "val1.jpg"  # 测试图像
    output_dir = "yolov8_ddawa_heatmaps_results"

    print(f"📁 模型路径: {model_path}")
    print(f"🖼️  图像路径: {image_path}")
    print(f"📂 输出目录: {output_dir}")

    # 创建可视化器
    visualizer = YOLOv8DDAWAHeatmapVisualizer(model_path)

    # 生成热力图
    visualizer.generate_heatmaps(image_path, output_dir)


    print("🎉 分析完成！")
