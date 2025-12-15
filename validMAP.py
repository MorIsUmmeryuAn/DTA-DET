# valid_single_image.py
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultralytics import YOLO
import cv2
from PIL import Image


def validate_single_image():
    # 加载训练好的模型
    model_path = '.pt'
    model = YOLO(model_path)

    # 选择验证集中的一张图像
    image_path = 'comparison1.jpg'  # 替换为实际路径 8和11没测过

    # 单张图像检测
    results = model(image_path, conf=0.65)

    # 显示结果
    for result in results:
        # 绘制检测框
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        im.save('single_image_validation.jpg')

        # 打印检测信息
        print(f"检测到 {len(result.boxes)} 个目标")
        for box in result.boxes:
            print(f"类别: {model.names[int(box.cls)]}, 置信度: {box.conf:.2f}")


if __name__ == '__main__':

    validate_single_image()
