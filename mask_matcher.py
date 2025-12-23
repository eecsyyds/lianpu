import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from fer import FER
import os

# ================= 配置区域 =================
INPUT_VIDEO_PATH = 'kun_clip.mp4'
CSV_PATH = 'tag.csv'
RETRIGGER_THRESHOLD = 9  # 未检测到人脸重新触发的阈值帧数
# ===========================================


class MaskMatcher:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='utf-8')
        self.df = self.df.fillna('无')

        # 初始化模型
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.fer_detector = FER(mtcnn=False)

        # --- 状态追踪变量 ---
        self.missing_face_count = RETRIGGER_THRESHOLD
        self.current_id = "1"
        self.current_emo = "neutral"
        self.current_params = {}

    def analyze_face_features(self, landmarks, emotion):
        """计算几何特征并映射参数"""
        h = np.linalg.norm(landmarks[10] - landmarks[152])
        w = np.linalg.norm(landmarks[234] - landmarks[454])
        face_ratio = h / w

        brow_vec = landmarks[105] - landmarks[70]
        brow_angle = np.degrees(np.arctan2(-brow_vec[1], brow_vec[0]))

        params = {}
        # 谱式映射逻辑
        if face_ratio > 1.35:
            params['谱式'] = "三块瓦脸"
        elif face_ratio < 1.15:
            params['谱式'] = "十字门脸"
        else:
            params['谱式'] = "整脸"

        # 眉形映射逻辑
        if brow_angle > 18:
            params['眉形'] = "刀眉"
        elif brow_angle > 8:
            params['眉形'] = "钩眉"
        else:
            params['眉形'] = "直眉"

        # 情绪颜色映射
        emo_color_map = {
            "happy": "红色", "angry": "黑色", "sad": "白色",
            "surprise": "黄色", "fear": "蓝色", "disgust": "绿色", "neutral": "粉色"
        }
        params['主色'] = emo_color_map.get(emotion, "红色")
        return params

    def get_top_matches(self, target_params, top_n=5):
        """计算得分并返回排名前N的脸谱"""
        results = []

        for _, row in self.df.iterrows():
            score = 0
            # 颜色匹配 (权重 3)
            if target_params['主色'] in str(row['主色']):
                score += 3
            # 谱式匹配 (权重 2)
            if target_params['谱式'] == str(row['谱式']):
                score += 2
            # 眉形匹配 (权重 1)
            if target_params['眉形'] in str(row['眉形']):
                score += 1

            results.append({
                'id': str(row['标识号']).strip(),
                'score': score,
                'style': row['谱式'],
                'color': row['主色'],
                'character': row['性格特征']
            })

        # 按得分从高到低排序，得分相同按ID排序
        sorted_results = sorted(
            results, key=lambda x: x['score'], reverse=True)
        return sorted_results[:top_n]


def main():
    matcher = MaskMatcher(CSV_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    frame_count = 0

    print(f"系统启动。消失阈值: {RETRIGGER_THRESHOLD} 帧")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = matcher.mp_face_mesh.process(rgb_frame)
        face_present = results.multi_face_landmarks is not None

        if face_present:
            if matcher.missing_face_count >= RETRIGGER_THRESHOLD:
                print(f"\n[触发检测] 帧数: {frame_count}")

                # A. 提取关键点
                h, w, _ = frame.shape
                landmarks = np.array([[int(lm.x * w), int(lm.y * h)]
                                     for lm in results.multi_face_landmarks[0].landmark])

                # B. 情绪检测
                emo_res = matcher.fer_detector.detect_emotions(frame)
                matcher.current_emo = "neutral"
                if emo_res:
                    matcher.current_emo = max(
                        emo_res[0]["emotions"], key=emo_res[0]["emotions"].get)

                # C. 获取匹配参数
                matcher.current_params = matcher.analyze_face_features(
                    landmarks, matcher.current_emo)

                # D. 计算前5名
                top_matches = matcher.get_top_matches(
                    matcher.current_params, top_n=5)

                # 更新当前锁定的 ID (排名第一的)
                matcher.current_id = top_matches[0]['id']

                # E. 输出排名前5
                print(
                    f"当前分析：情绪={matcher.current_emo}, 目标主色={matcher.current_params['主色']}, 目标谱式={matcher.current_params['谱式']}")
                print("-" * 60)
                print(f"{'排名':^4} | {'标识号':^6} | {'得分':^4} | {'谱式':^10} | {'性格特征'}")
                for i, res in enumerate(top_matches, 1):
                    print(
                        f" {i:^4} | {res['id']:^8} | {res['score']:^4} | {res['style']:^12} | {res['character'][:20]}")
                print("-" * 60)

            matcher.missing_face_count = 0
        else:
            matcher.missing_face_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
