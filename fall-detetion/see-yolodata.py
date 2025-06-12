import os
import random
import cv2
import time
 
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
             'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
             'left_ankle', 'right_ankle']
 
 
def load_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        label = int(parts[0])
        bbox = list(map(float, parts[1:]))
        labels.append((label, bbox))
    return labels
 
 
def draw_keypoints(image, keypoints, height, width, color=((0, 255, 0), (0, 0, 255))):
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] * width
        y = keypoints[i + 1] * height
        v = int(keypoints[i + 2])
 
        if x > 0 or y > 0:  # 忽略无效点
            cv2.circle(image, (int(x), int(y)), 3, color[v % 2], -1)
 
 
def draw_boxes(image, labels, label_map, colors):
    height, width, _ = image.shape
    for label, bbox in labels:
        x_center, y_center, w, h = bbox[:4]
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
        color = colors[label % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image, str(label_map[label]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
 
        draw_keypoints(image, bbox[4:], height, width)
 
 
def show_image(image_path, label_path, label_map, colors):
    print(image_path)
    image = cv2.imread(image_path)
    if os.path.exists(label_path):
        labels = load_labels(label_path)
        draw_boxes(image, labels, label_map, colors)
 
    cv2.imshow('YOLO Dataset', image)
 
 
def visualize_yolo_dataset(images_path, labels_path, label_map, n, auto=False, interval=1.0, seed=None):
    """
    根据图片和标签显示带检测框的图片。按‘q’退出显示。
    :param images_path: 图片路径
    :param labels_path: 标签路径
    :param label_map: 将标签值映射为标签名
    :param n: 显示的图片数
    :param auto: False按任意非‘q’跳到下一张；True自动播放
    :param interval: 自动播放的时间间隔
    :param seed: 显示随机顺序的随机种子
    :return:
    """
    image_files = os.listdir(images_path)
 
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(time.time())
    random.shuffle(image_files)
 
    idx = 0
    while idx < min(n, len(image_files)):
        image_file = image_files[idx]
        image_path = os.path.join(images_path, image_file)
        label_path = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.txt')
 
        show_image(image_path, label_path, label_map, colors)
 
        if auto:
            interval_ms = int(interval * 1000)
            key = cv2.waitKey(interval_ms) & 0xFF  # Wait for interval seconds
        else:
            key = cv2.waitKey(0) & 0xFF  # Wait for a key press
 
        if key == ord('q'):
            break
        idx += 1
 
    cv2.destroyAllWindows()
 
 
if __name__ == '__main__':
    # 示例标签映射字典
    # label_map = {
    #     0: 'ycj',
    #     1: 'kx',
    #     2: 'kx_dk',
    #     3: 'money',
    #     4: 'zbm',
    # }
    label_map = {k: k for k in range(80)}
    # 数据集路径
    dataset_path = f'./coco2017/pose_ren'
    images_path = os.path.join(dataset_path, 'images/val2017')
    labels_path = os.path.join(dataset_path, 'labels/val2017')
 
    visualize_yolo_dataset(images_path, labels_path, label_map, n=1e6, auto=False, interval=2.5, seed=2024)