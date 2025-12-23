import cv2
import numpy as np
import mediapipe as mp
import os
import time
from fer import FER


# ================= 配置区域 =================
INPUT_VIDEO_PATH = 'kun_clip.mp4'      # 输入视频路径
DIR = 'masks'
PROXY_DIR = 'proxies'
# mediapip 不能识别京剧脸谱，需要一个和其五官对齐的真实人脸以获得关键点
OUTPUT_VIDEO_PATH = 'output2.mp4'  # 结果保存路径

# 额头高度倍率
# 原始的 mediapipe 关键点只到额头，手动上移以覆盖更多区域
FOREHEAD_HEIGHT = 1.4
# ===========================================


def get_mediapipe_landmarks(image, face_mesh):
    """提取 MediaPipe 468个关键点"""
    h, w, c = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0]
    points = []
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def apply_affine_transform(src, src_tri, dst_tri, size):
    """仿射变换"""
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None,
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst


def extend_forehead_height(landmarks, scale=1.3):
    """拉高额头区域"""
    points = landmarks.copy()

    # 33: 左眼角, 263: 右眼角
    left_eye_inner = points[33]
    right_eye_inner = points[263]
    eyebrow_line_y = (left_eye_inner[1] + right_eye_inner[1]) // 2

    for i in range(len(points)):
        x, y = points[i]
        if y < eyebrow_line_y:
            dist_from_eyebrow = eyebrow_line_y - y
            new_dist = dist_from_eyebrow * scale
            points[i][1] = int(eyebrow_line_y - new_dist)
    return points


def get_triangles_indices(points_proxy, width, height):
    """
    基于替身（Proxy）计算三角网格索引。
    """
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)

    # 插入替身的关键点
    for p in points_proxy:
        x = min(max(0, int(p[0])), width - 1)
        y = min(max(0, int(p[1])), height - 1)
        subdiv.insert((x, y))

    triangles = subdiv.getTriangleList()
    triangles_indices = []

    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        def get_index(pt, points_array):
            dists = np.linalg.norm(points_array - np.array(pt), axis=1)
            min_dist_idx = np.argmin(dists)
            if dists[min_dist_idx] < 2.0:
                return min_dist_idx
            return -1

        index1 = get_index(pt1, points_proxy)
        index2 = get_index(pt2, points_proxy)
        index3 = get_index(pt3, points_proxy)

        if index1 != -1 and index2 != -1 and index3 != -1:
            triangles_indices.append((index1, index2, index3))

    return triangles_indices


def main():
    # 1. 检查文件
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"错误：找不到视频文件 {INPUT_VIDEO_PATH}")
        return

    # 2. 初始化 MediaPipe
    # 注意：static_image_mode=False 开启视频流模式，更稳更快
    print("正在初始化 MediaPipe...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 初始化 FER（用于情绪检测）
    fer_detector = FER(mtcnn=False)

    # 3. 预处理静态资源 (只做一次)
    print("正在预处理静态素材（7种情绪）...")

    PROXY_DIR_LOCAL = PROXY_DIR
    MASK_DIR_LOCAL = DIR

    # DeepFace 的 7 类情绪标签
    EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    # 缓存：每种情绪一套资源（mask、proxy点、三角网格）
    mask_rgba = {}
    points_proxy_map = {}
    triangles_map = {}

    # 先读一张 mask 获取输出尺寸（用 happy.png 也行）
    probe = cv2.imread(os.path.join(MASK_DIR_LOCAL, "happy.png"), cv2.IMREAD_UNCHANGED)
    if probe is None:
        print("错误：读取 masks/happy.png 失败")
        face_mesh.close()
        return
    base_h, base_w = probe.shape[:2]

    for emo in EMOTIONS:
        mask_path = os.path.join(MASK_DIR_LOCAL, f"{emo}.png")
        proxy_path = os.path.join(PROXY_DIR_LOCAL, f"{emo}_face.png")

        if not os.path.exists(mask_path):
            print(f"错误：缺少脸谱 {mask_path}")
            face_mesh.close()
            return
        if not os.path.exists(proxy_path):
            print(f"错误：缺少 proxy {proxy_path}")
            face_mesh.close()
            return

        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        p = cv2.imread(proxy_path)  # proxy 不需要 alpha

        if m is None or p is None:
            print(f"错误：读取失败 {mask_path} 或 {proxy_path}")
            face_mesh.close()
            return

        # 统一尺寸：mask 和 proxy 都 resize 到 base 尺寸（非常重要）
        m = cv2.resize(m, (base_w, base_h), interpolation=cv2.INTER_AREA)
        p = cv2.resize(p, (base_w, base_h), interpolation=cv2.INTER_AREA)

        # 保证 mask 有 alpha 通道
        if m.ndim == 2:
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGRA)
        if m.shape[2] == 3:
            b, g, r = cv2.split(m)
            a = np.ones_like(b) * 255
            m = cv2.merge((b, g, r, a))

        # 用 mediapipe 在 proxy 上提点（这就是你 proxies 的意义）
        pts = get_mediapipe_landmarks(p, face_mesh)
        if pts is None:
            print(f"错误：proxy 中未检测到人脸：{proxy_path}")
            face_mesh.close()
            return

        tris = get_triangles_indices(pts, base_w, base_h)

        mask_rgba[emo] = m
        points_proxy_map[emo] = pts
        triangles_map[emo] = tris

    print("7种情绪素材加载完成：", {k: len(v) for k, v in triangles_map.items()})

    # 4. 视频设置
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {INPUT_VIDEO_PATH}")
        face_mesh.close()
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None or fps <= 1e-3 or np.isnan(fps):
        fps = 30.0

    # 结果写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    if not out.isOpened():
        print("错误：VideoWriter 打开失败。可以尝试改 fourcc 为 'XVID' 输出 AVI。")
        cap.release()
        face_mesh.close()
        return

    print(f"开始处理视频: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

    frame_count = 0
    start_time = time.time()

    last_emotion = "neutral"
    last_emotion_score = 0.0
    EMO_EVERY_N_FRAMES = 10 #几帧调用一次情绪检测

    # 5. 逐帧循环
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"正在处理帧: {frame_count}/{total_frames} ...")

        # 检测当前帧关键点
        # 注意：这里不需要cvtColor，因为get_mediapipe_landmarks里做了，
        # 但为了性能，其实可以在这里转一次RGB，避免函数内重复转，这里维持原样即可。
        points_target_orig = get_mediapipe_landmarks(frame, face_mesh)

        # 如果这一帧没检测到脸，直接写入原图，继续下一帧
        if points_target_orig is None:
            out.write(frame)
            continue

        # ===== DeepFace 情绪检测（每N帧一次）=====
        if frame_count % EMO_EVERY_N_FRAMES == 0:
            xs = points_target_orig[:, 0]
            ys = points_target_orig[:, 1]
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            pad = 20
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(width - 1, x2 + pad); y2 = min(height - 1, y2 + pad)

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size > 0:
                try:
                    # FER 使用 BGR 输入即可
                    emo_res = fer_detector.detect_emotions(face_crop)
                    if emo_res and isinstance(emo_res, list) and "emotions" in emo_res[0]:
                        emo_scores = emo_res[0]["emotions"]
                        # FER 输出的 key 就是 angry/disgust/fear/happy/sad/surprise/neutral
                        last_emotion = max(emo_scores, key=emo_scores.get)
                        last_emotion_score = float(emo_scores.get(last_emotion, 0.0))
                except Exception:
                    pass

        # DeepFace 可能输出不在7类里（极少），做兜底
        if last_emotion not in mask_rgba:
            last_emotion = "neutral"
        # =======================================

        # 拉伸额头
        points_target_stretched = extend_forehead_height(
            points_target_orig, scale=FOREHEAD_HEIGHT)

        # 准备画布
        img_warped_mask = np.zeros((height, width, 4), dtype=np.uint8)

        # ===== 根据情绪选择当前脸谱与proxy拓扑 =====
        img_mask_src = mask_rgba[last_emotion]
        points_proxy = points_proxy_map[last_emotion]
        triangles_indices = triangles_map[last_emotion]
        # =========================================

        # 遍历三角形并渲染
        # (这里的逻辑和单张图片一样，但是索引 indices 是复用的)
        for tri_index in triangles_indices:
            # 源：替身点
            tr1_src = points_proxy[tri_index[0]]
            tr2_src = points_proxy[tri_index[1]]
            tr3_src = points_proxy[tri_index[2]]

            # 目标：视频当前帧的点 (拉伸后)
            tr1_tgt = points_target_stretched[tri_index[0]]
            tr2_tgt = points_target_stretched[tri_index[1]]
            tr3_tgt = points_target_stretched[tri_index[2]]

            rect_src = cv2.boundingRect(
                np.float32([tr1_src, tr2_src, tr3_src]))
            rect_tgt = cv2.boundingRect(
                np.float32([tr1_tgt, tr2_tgt, tr3_tgt]))

            (x, y, w, h) = rect_src
            (x_t, y_t, w_t, h_t) = rect_tgt

            if w <= 0 or h <= 0 or w_t <= 0 or h_t <= 0:
                continue

            # 边界检查
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > img_mask_src.shape[1]:
                w = img_mask_src.shape[1] - x
            if y + h > img_mask_src.shape[0]:
                h = img_mask_src.shape[0] - y
            if w <= 0 or h <= 0:
                continue

            # ===== 目标rect_tgt边界裁剪（非常重要）=====
            if x_t < 0:
                x_t = 0
            if y_t < 0:
                y_t = 0
            if x_t + w_t > width:
                w_t = width - x_t
            if y_t + h_t > height:
                h_t = height - y_t
            if w_t <= 0 or h_t <= 0:
                continue
            # ==========================================

            crop_src = img_mask_src[y:y+h, x:x+w]

            points_src_crop = np.float32([[tr1_src[0]-x, tr1_src[1]-y],
                                          [tr2_src[0]-x, tr2_src[1]-y],
                                          [tr3_src[0]-x, tr3_src[1]-y]])
            points_tgt_crop = np.float32([[tr1_tgt[0]-x_t, tr1_tgt[1]-y_t],
                                          [tr2_tgt[0]-x_t, tr2_tgt[1]-y_t],
                                          [tr3_tgt[0]-x_t, tr3_tgt[1]-y_t]])

            warped_crop = apply_affine_transform(
                crop_src, points_src_crop, points_tgt_crop, (w_t, h_t))

            mask_tri = np.zeros((h_t, w_t), dtype=np.uint8)
            cv2.fillConvexPoly(mask_tri, np.int32(points_tgt_crop), 1)

            img_warped_mask_rect = img_warped_mask[y_t:y_t+h_t, x_t:x_t+w_t]

            if img_warped_mask_rect.shape[:2] == warped_crop.shape[:2]:
                img_warped_mask_rect[mask_tri ==
                                     1] = warped_crop[mask_tri == 1]

        # 挖洞
        def get_contour_points(indices, landmarks):
            return np.array([landmarks[i] for i in indices], dtype=np.int32)

        LEFT_EYE = [33, 246, 161, 160, 159, 158, 157,
                    173, 133, 155, 154, 153, 145, 144, 163, 7]
        RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388,
                     466, 263, 249, 390, 373, 374, 380, 381, 382]
        LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310,
                      415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        alpha_channel = img_warped_mask[:, :, 3].copy()
        cv2.fillPoly(alpha_channel, [get_contour_points(
            LEFT_EYE, points_target_stretched)], 0)
        cv2.fillPoly(alpha_channel, [get_contour_points(
            RIGHT_EYE, points_target_stretched)], 0)
        cv2.fillPoly(alpha_channel, [get_contour_points(
            LIPS_INNER, points_target_stretched)], 0)

        alpha_channel = cv2.GaussianBlur(alpha_channel, (5, 5), 0)
        img_warped_mask[:, :, 3] = alpha_channel

        # 合成
        img_final = frame.astype(float)
        alpha = img_warped_mask[:, :, 3].astype(float) / 255.0
        mask_bgr = img_warped_mask[:, :, 0:3].astype(float)
        alpha = cv2.merge([alpha, alpha, alpha])

        img_final = mask_bgr * alpha + img_final * (1.0 - alpha)
        img_final = img_final.astype(np.uint8)

        cv2.putText(img_final, f"{last_emotion} ({last_emotion_score:.2f})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 写入视频帧
        out.write(img_final)

    # 6. 清理
    cap.release()
    out.release()
    face_mesh.close()

    end_time = time.time()
    print(f"处理完成！耗时: {end_time - start_time:.2f}秒")
    print(f"结果已保存至: {os.path.abspath(OUTPUT_VIDEO_PATH)}")


if __name__ == '__main__':
    main()
