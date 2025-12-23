import cv2
import numpy as np
import mediapipe as mp
import os
import random
from fer import FER

from mask_matcher import MaskMatcher

# ================= 配置区域 =================
INPUT_VIDEO_PATH = 'kun_clip.mp4'
OUTPUT_VIDEO_PATH = 'output_random.mp4'
CSV_PATH = 'tag.csv'
IMAGES_DIR = 'images'       # 自动匹配模式下的脸谱目录 (lianpu_ID.jpg)
MASK_FIXED_DIR = 'masks'    # 固定模式下的脸谱目录 (happy.png...)
PROXY_DIR = 'proxies'       # 替身图片目录

# --- 功能开关 ---
USE_AUTO_MATCH = True        # True: 使用 CSV 特征匹配; False: 使用固定情绪名称
ENABLE_EYE_MOUTH_CROP = False  # True: 裁剪(挖掉)眼睛和嘴巴; False: 全脸覆盖

# --- 参数微调 ---
RETRIGGER_THRESHOLD = 9      # 重新触发匹配的帧数阈值
FOREHEAD_HEIGHT = 1.4        # 额头拉伸倍率
# ===========================================


def get_mediapipe_landmarks(image, face_mesh):
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    points = [[int(lm.x * w), int(lm.y * h)]
              for lm in results.multi_face_landmarks[0].landmark]
    return np.array(points, dtype=np.int32)


def extend_forehead_height(landmarks, scale=1.3):
    points = landmarks.copy()
    # 33: 左眼内角, 263: 右眼内角 (用于确定眉线基准)
    eyebrow_line_y = (points[33][1] + points[263][1]) // 2
    for i in range(len(points)):
        if points[i][1] < eyebrow_line_y:
            points[i][1] = int(
                eyebrow_line_y - (eyebrow_line_y - points[i][1]) * scale)
    return points


def get_triangles_indices(points_proxy, width, height):
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for p in points_proxy:
        subdiv.insert((float(p[0]), float(p[1])))
    triangles = subdiv.getTriangleList()
    indices = []
    for t in triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for p in pts:
            dists = np.linalg.norm(points_proxy - np.array(p), axis=1)
            idx.append(np.argmin(dists))
        indices.append(idx)
    return indices


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def main():
    # 初始化
    matcher = MaskMatcher(CSV_PATH)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    fer_detector = FER(mtcnn=False)

    # 获取基准 Proxy
    proxy_files = sorted([f for f in os.listdir(
        PROXY_DIR) if f.endswith(('.png', '.jpg'))])
    if not proxy_files:
        print("错误：Proxies 目录为空")
        return

    proxy_img = cv2.imread(os.path.join(PROXY_DIR, proxy_files[0]))
    ph, pw = proxy_img.shape[:2]
    points_proxy = get_mediapipe_landmarks(proxy_img, mp_face_mesh)
    triangles_indices = get_triangles_indices(points_proxy, pw, ph)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (width, height))

    missing_face_count = RETRIGGER_THRESHOLD
    current_mask_img = None
    current_resource_key = None
    current_emo = "neutral"

    print(f"系统就绪。随机选择模式: {'开启' if USE_AUTO_MATCH else '关闭'}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks_raw = get_mediapipe_landmarks(frame, mp_face_mesh)

        if landmarks_raw is not None:
            if missing_face_count >= RETRIGGER_THRESHOLD:
                # A. 情绪识别
                emo_res = fer_detector.detect_emotions(frame)
                current_emo = max(
                    emo_res[0]["emotions"], key=emo_res[0]["emotions"].get) if emo_res else "neutral"

                mask_path = ""
                new_resource_key = None

                if USE_AUTO_MATCH:
                    # B. 自动匹配 + 2. 随机选择前五
                    params = matcher.analyze_face_features(
                        landmarks_raw, current_emo)
                    top_matches = matcher.get_top_matches(params, top_n=5)

                    if top_matches:
                        selected = random.choice(top_matches)  # 随机选择
                        new_resource_key = selected['id']
                        mask_path = os.path.join(
                            IMAGES_DIR, f"lianpu_{new_resource_key}.jpg")
                        if not os.path.exists(mask_path):
                            mask_path = mask_path.replace('.jpg', '.png')
                else:
                    # 固定模式
                    new_resource_key = current_emo
                    mask_path = os.path.join(
                        MASK_FIXED_DIR, f"{new_resource_key}.png")
                    if not os.path.exists(mask_path):
                        mask_path = mask_path.replace('.png', '.jpg')

                # C. 加载资源
                if new_resource_key != current_resource_key:
                    temp_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if temp_img is not None:
                        current_mask_img = cv2.resize(temp_img, (pw, ph))
                        if current_mask_img.shape[2] == 3:
                            current_mask_img = cv2.cvtColor(
                                current_mask_img, cv2.COLOR_BGR2BGRA)
                        current_resource_key = new_resource_key
                        print(
                            f"[匹配更新] Key: {new_resource_key}, 情绪: {current_emo}")

            missing_face_count = 0

            # --- 渲染逻辑 (包含边界修复) ---
            points_target = extend_forehead_height(
                landmarks_raw, scale=FOREHEAD_HEIGHT)
            img_warped = np.zeros((height, width, 4), dtype=np.uint8)

            for tri in triangles_indices:
                pts_src, pts_dst = points_proxy[tri], points_target[tri]
                rect_d = cv2.boundingRect(np.float32([pts_dst]))
                xd, yd, wd, hd = rect_d

                x1, y1 = max(0, xd), max(0, yd)
                x2, y2 = min(width, xd + wd), min(height, yd + hd)
                if x1 >= x2 or y1 >= y2:
                    continue

                offset_x, offset_y = x1 - xd, y1 - yd
                actual_w, actual_h = x2 - x1, y2 - y1

                rect_s = cv2.boundingRect(np.float32([pts_src]))
                xs, ys, ws, hs = rect_s
                crop_s = current_mask_img[ys:ys+hs, xs:xs+ws]
                pts_s_c = np.float32([[p[0]-xs, p[1]-ys] for p in pts_src])
                pts_d_c = np.float32([[p[0]-xd, p[1]-yd] for p in pts_dst])

                warped_tile = apply_affine_transform(
                    crop_s, pts_s_c, pts_d_c, (wd, hd))
                mask_tri = np.zeros((hd, wd), dtype=np.uint8)
                cv2.fillConvexPoly(mask_tri, np.int32(pts_d_c), 1)

                # 安全融合
                target_slice = img_warped[y1:y2, x1:x2]
                mask_slice = mask_tri[offset_y:offset_y +
                                      actual_h, offset_x:offset_x+actual_w]
                tile_slice = warped_tile[offset_y:offset_y +
                                         actual_h, offset_x:offset_x+actual_w]
                target_slice[mask_slice == 1] = tile_slice[mask_slice == 1]

            # --- 五官裁剪 ---
            alpha = img_warped[:, :, 3].copy()
            if ENABLE_EYE_MOUTH_CROP:
                def get_pts(indices): return np.array(
                    [points_target[i] for i in indices], np.int32)
                # 眼睛和嘴巴内圈
                cv2.fillPoly(
                    alpha, [get_pts([33, 160, 158, 133, 153, 144])], 0)
                cv2.fillPoly(
                    alpha, [get_pts([362, 385, 387, 263, 373, 380])], 0)
                cv2.fillPoly(
                    alpha, [get_pts([78, 81, 13, 311, 308, 402, 14, 178])], 0)

            alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

            # 合成
            foreground = img_warped[:, :, :3].astype(float)
            background = frame.astype(float)
            alpha_map = cv2.merge([alpha, alpha, alpha]).astype(float) / 255.0
            frame = (foreground * alpha_map + background *
                     (1.0 - alpha_map)).astype(np.uint8)

            cv2.putText(frame, f"ID: {current_resource_key} ({current_emo})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            missing_face_count += 1

        out.write(frame)

    cap.release()
    out.release()
    print("视频生成完毕。")


if __name__ == "__main__":
    main()
