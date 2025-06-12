import cv2
import time
import os
from tracker import BYTETracker, Object
from pose import Yolov8Pose # 确保导入 fall_estimate 函数

# 假设你已经有了以下类和函数的Python实现
# from pose import Yolov8Pose, SKELETON, KPS_COLORS, LIMB_COLORS

# 模型路径（需要根据你的实际路径修改）
YOLOV8_PT = "../runs/pose/train/weights/best.pt"
SAVE_PATH = "../outputs"

# 创建模型实例
yolov8_pose = Yolov8Pose(YOLOV8_PT)

def process_camera(camera_index: int = 0):
    """
    处理摄像头实时输入
    
    参数:
        camera_index: 摄像头索引号，默认为 0（通常表示默认摄像头）
        
    返回:
        None
    """
    # 打开摄像头
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"无法打开摄像头 (索引: {camera_index})")
        return

    # 获取摄像头属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"摄像头分辨率: {frame_width}x{frame_height}, FPS: {fps}")

    # 初始化跟踪器
    tracker = BYTETracker(frame_rate=fps, track_buffer=50)

    frame_count = 0
    start_time = time.time()

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 目标检测
        results = yolov8_pose.detect_yolov8(frame)
        
        # 跌倒检测和跟踪
        if results:
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy().tolist()[0]
                    width = x2 - x1
                    height = y2 - y1
                    prob = box.conf.cpu().numpy().tolist()[0]
                    cls_id = box.cls.cpu().numpy().tolist()[0]
                    
                    if cls_id == 0:  # 假设类别0是人
                        detections.append(Object((x1, y1, width, height), prob))

                # 更新跟踪器
                tracks = tracker.update(detections)

                # 绘制跟踪结果并进行跌倒检测
                for track in tracks:
                    tlbr = track.tlbr
                    track_id = track.track_id
                    frame = result[0].plot()
                    # cv2.rectangle(frame, (int(tlbr[0]), int(tlbr[1])), (int(tlbr[2]), int(tlbr[3])), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (int(tlbr[0]), int(tlbr[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 获取关键点
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy().tolist()[0]
                        confidences = result.keypoints.conf.cpu().numpy().tolist()[0]
                        fall = yolov8_pose.fall_estimate(keypoints, confidences)
                        if fall:
                            cv2.putText(frame, 'Fall Detected', (int(tlbr[0]), int(tlbr[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 显示图像
        cv2.namedWindow("Camera Detection Result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Detection Result", 1980, 1080)
        result = cv2.resize(frame, (1980, 1080), interpolation=cv2.INTER_AREA)
        cv2.imshow("Camera Detection Result", result)

        # 按 'q' 键退出
        if cv2.waitKey(1) == ord('q'):
            break

    end_time = time.time()

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 输出处理信息
    print(f"处理完成！总帧数: {frame_count}")
    print(f"平均FPS: {frame_count / (end_time - start_time):.2f}")

def main():
    """
    主函数：处理命令行参数并调用相应的处理函数
    
    参数:
        None
        
    返回:
        None
    """
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLOv8-Pose人体姿态估计')
    parser.add_argument('input_type', type=str, help='输入类型: image, video 或 camera')
    parser.add_argument('input_path', type=str, nargs='?', default="0", help='输入文件路径或摄像头索引（仅限 camera 类型）')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # 根据输入类型处理图片、视频或摄像头
    if args.input_type == 'image':
        # process_image(args.input_path)
        pass
    elif args.input_type == 'video':
        # process_video(args.input_path)
        pass
    elif args.input_type == 'camera':
        try:
            camera_index = int(args.input_path)
        except ValueError:
            print("摄像头索引必须是整数！")
            exit(-1)
        process_camera(camera_index)
    else:
        print("无效的输入类型。请使用 'image'、'video' 或 'camera'.")
        exit(-1)


if __name__ == '__main__':
    main()