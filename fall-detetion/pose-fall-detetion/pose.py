from ultralytics import YOLO
import math

# 定义关键点颜色
KPS_COLORS = [
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255]
]

# 定义骨架连接
SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7]
]

# 定义肢体颜色
LIMB_COLORS = [
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [51, 153, 255],
    [255, 51, 255],
    [255, 51, 255],
    [255, 51, 255],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [255, 128, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0]
]


class Yolov8Pose:
    def __init__(self, model_path):
        """初始化YOLOv8姿态估计模型"""
        self.model = YOLO(model_path)
    
    def detect_yolov8(self, frame):
        model = self.model(frame)
        return model
    
    def fall_estimate(self, kps, kpc):
        is_fall = False

        # 1. 先获取哪些用于判断的点坐标
        L_shoulder = kps[5]  # 左肩
        L_shoulder_confi = kpc[5]
        R_shoulder = kps[6]  # 右肩
        R_shoulder_confi = kpc[6]
        C_shoulder = ((L_shoulder[0] + R_shoulder[0]) // 2, (L_shoulder[1] + R_shoulder[1]) // 2)  # 肩部中点

        L_hip = kps[11]  # 左髋
        L_hip_confi = kpc[11]
        R_hip = kps[12] # 右髋
        R_hip_confi = kpc[12]
        C_hip = ((L_hip[0] + R_hip[0]) // 2, (L_hip[1] + R_hip[1]) // 2)  # 髋部中点

        L_knee = kps[13]  # 左膝
        L_knee_confi = kpc[13]
        R_knee = kps[14] # 右膝
        R_knee_confi = kpc[14]
        C_knee = ((L_knee[0] + R_knee[0]) // 2, (L_knee[1] + R_knee[1]) // 2)  # 膝部中点

        L_ankle = kps[15] # 左踝
        L_ankle_confi = kpc[15]
        R_ankle = kps[16]  # 右踝
        R_ankle_confi = kpc[16]
        C_ankle = ((L_ankle[0] + R_ankle[0]) // 2, (L_ankle[1] + R_ankle[1]) // 2)  # 计算脚踝中点

        # 2. 第一个判定条件： 若肩的纵坐标最小值min(L_shoulder.y, R_shoulder.y)不低于脚踝的中心点的纵坐标C_ankle.y
        # 且p_shoulders、p_ankle关键点置信度大于预设的阈值，则疑似摔倒。
        if (L_shoulder_confi > 0.0 and R_shoulder_confi > 0.0 and L_ankle_confi > 0.0 and R_ankle_confi > 0.0 and 
            L_shoulder[1] > 0.0 and R_shoulder[1] > 0.0 and L_ankle[1] > 0.0 and R_ankle[1] > 0.0):
            shoulder_y_min = min(L_shoulder[1], R_shoulder[1])
            if shoulder_y_min >= C_ankle[1]:
                is_fall = True
                return is_fall

        # 3. 第二个判断条件：若肩的纵坐标最大值max(L_shoulder.y, R_shoulder.y)大于膝盖纵坐标的最小值min(L_knee.y, R_knee.y)，
        # 且p_shoulders、p_knees关键点置信度大于预设的阈值，则疑似摔倒。
        if (L_shoulder_confi > 0.0 and R_shoulder_confi > 0.0 and L_knee_confi > 0.0 and R_knee_confi > 0.0 and
            L_shoulder[1] > 0.0 and R_shoulder[1] > 0.0 and L_knee[1] > 0.0 and R_knee[1] > 0.0):
            shoulder_y_max = max(L_shoulder[1], R_shoulder[1])
            knee_y_min = min(L_knee[1], R_knee[1])
            if shoulder_y_max > knee_y_min:
                is_fall = True
                return is_fall

        # 4, 第三个判断条件：计算关键点最小外接矩形的宽高比。p0～p16在x方向的距离是xmax-xmin，在方向的距离是ymax-ymin，
        # 若(xmax-xmin) / (ymax-ymin)不大于指定的比例阈值，则判定为未摔倒，不再进行后续判定。
        
        if C_shoulder[0] != 0.0 and C_shoulder[1] != 0.0 and C_ankle[0] != 0.0 and C_ankle[1] != 0.0: 
        
            num_point = 17  # 17个关键点

            xmin = float('inf')
            ymin = float('inf')
            xmax = -float('inf')
            ymax = -float('inf')

            for k in range(num_point):
                if k < num_point:
                    kps_x = round(kps[k][0])  # 关键点x
                    kps_y = round(kps[k][1])  # 关键点y
                    kps_c = kpc[k]  # 可信性

                    if kps_c > 0.0:
                        xmin = min(xmin, kps_x)
                        xmax = max(xmax, kps_x)
                        ymin = min(ymin, kps_y)
                        ymax = max(ymax, kps_y)

            # 检查是否存在有效的宽度和高度
            if xmax > xmin and ymax > ymin:
                aspect_ratio = (xmax - xmin) / (ymax - ymin)

                # 如果宽高比大于指定阈值，则判定为摔倒
                if aspect_ratio > 2.00:
                    is_fall = True
                    return is_fall

        # 5. 第四个判断条件：通过两膝与髋部中心点的连线与地面的夹角判断。首先假定有两点p1＝(x1 ,y1 )，p2＝(x2 ,y2 )，那么两点连接线与地面的角度计算公式为：
        # 												θ = arctan((y2-y1) / (x2-x1)) * 180 / pi
        # 此处左膝与髋部的两点是(C_hip, L_knee)，与地面夹角表示为θ1；右膝与髋部的两点 是(C_hip, R_knee)，与地面夹角表示为θ2，
        # 若min(θ1 ,θ2 )＜th1 或 max(θ1 ,θ2 )＜th2，且p_knees、 p_hips关键点置信度大于预设的阈值，则疑似摔倒
        if L_knee_confi > 0.0 and R_knee_confi > 0.0 and L_hip_confi > 0.0 and R_hip_confi > 0.0:
            # 左膝与髋部中心的角度
            theta1 = math.atan2(L_knee[1] - C_hip[1], L_knee[0] - C_hip[0]) * 180.0 / math.pi
            # 右膝与髋部中心的角度
            theta2 = math.atan2(R_knee[1] - C_hip[1], R_knee[0] - C_hip[0]) * 180.0 / math.pi

            min_theta = min(abs(theta1), abs(theta2))
            max_theta = max(abs(theta1), abs(theta2))

            th1 = 30.0  # 假设的最小角度阈值
            th2 = 70.0  # 假设的最大角度阈值

            if (min_theta) < th1 and (max_theta > th2):
                is_fall = True
                return is_fall

        # 第五个判断条件：通过肩、髋部、膝盖夹角，髋部、膝盖、脚踝夹角判断。
        if (L_shoulder_confi > 0.0 and R_shoulder_confi > 0.0 and L_hip_confi > 0.0 and R_hip_confi > 0.0 and
                L_knee_confi > 0.0 and R_knee_confi > 0.0 and L_ankle_confi > 0.0 and R_ankle_confi > 0.0):
            # 计算向量 v1 和 v2
            v1 = (C_shoulder[0] - C_hip[0], C_shoulder[1] - C_hip[1])
            v2 = (C_knee[0] - C_hip[0], C_knee[1] - C_hip[1])

            # 计算向量 v3 和 v4
            v3 = (C_hip[0] - C_knee[0], C_hip[1] - C_knee[1])
            v4 = (C_ankle[0] - C_knee[0], C_ankle[1] - C_knee[1])

            # 计算向量 v1 和 v2 的夹角 θ3
            dot_product1 = v1[0] * v2[0] + v1[1] * v2[1]

            # 计算向量 v3 和 v4 的夹角 θ4
            dot_product2 = v3[0] * v4[0] + v3[1] * v4[1]

            if dot_product1 != 0.0 and dot_product2 != 0.0:
                magnitude1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2) * math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                theta3 = math.acos(dot_product1 / magnitude1) * 180.0 / math.pi

                magnitude2 = math.sqrt(v3[0] ** 2 + v3[1] ** 2) * math.sqrt(v4[0] ** 2 + v4[1] ** 2)
                theta4 = math.acos(dot_product2 / magnitude2) * 180.0 / math.pi

                th3 = 70.0  # 假设的阈值，肩、髋和膝的角度
                th4 = 30.0  # 假设的阈值，髋、膝和脚踝的角度

                if (theta3 < th3) and (theta4 < th4):
                    is_fall = True

        return is_fall