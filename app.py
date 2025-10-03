import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ======================
# フィルタ関数群
# ======================
def Deuteranope(im):
    b, g, r = cv2.split(im)
    bl = np.power((b / 255 + 0.055) / 1.055, 2.4)
    gl = np.power((g / 255 + 0.055) / 1.055, 2.4)
    rl = np.power((r / 255 + 0.055) / 1.055, 2.4)

    l = 0.31394 * rl + 0.63957 * gl + 0.04652 * bl
    m = 0.15530 * rl + 0.75796 * gl + 0.08673 * bl
    s = 0.01772 * rl + 0.10945 * gl + 0.87277 * bl

    m = np.where(s <= l, 0.82781 * l + 0.17216 * s,
                 0.81951 * l + 0.18046 * s)

    rl = 5.47213 * l - 4.64189 * m + 0.16958 * s
    gl = -1.12464 * l + 2.29255 * m - 0.16786 * s
    bl = 0.02993 * l - 0.19325 * m + 1.16339 * s

    bd = (np.power(bl, 1.0 / 2.4) * 1.055 - 0.055) * 255
    gd = (np.power(gl, 1.0 / 2.4) * 1.055 - 0.055) * 255
    rd = (np.power(rl, 1.0 / 2.4) * 1.055 - 0.055) * 255

    return cv2.merge((bd.clip(0, 255).astype(np.uint8),
                      gd.clip(0, 255).astype(np.uint8),
                      rd.clip(0, 255).astype(np.uint8)))


def dark(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -128 * (a - 133) / (164 - 133), 0)
    x = x.clip(-128, 0)
    l = cv2.add(l.astype(np.float64), x)
    l = l.clip(0, 255).astype(np.uint8)
    a = a.astype(np.uint8)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


def blue(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -31 * (a - 133) / (164 - 133), 0)
    x = x.clip(-31, 0)
    b = cv2.add(b.astype(np.float64), x)
    b = b.clip(0, 255).astype(np.uint8)
    a = a.astype(np.uint8)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


def yellow(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, 31 * (a - 133) / (164 - 133), 0)
    x = x.clip(0, 31)
    b = cv2.add(b.astype(np.float64), x)
    b = b.clip(0, 255).astype(np.uint8)
    a = a.astype(np.uint8)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


ls = 6
def lattice(i, j):
    return (i // ls + j // ls) % 2 == 0


def darkandbright(im):
    w, h, _ = im.shape
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    l = l.astype(np.float64)
    a = a.astype(np.float64)
    for j in range(h):
        for i in range(w):
            if a[i, j] >= 133:
                x = 31 * (a[i, j] - 133) / (164 - 133)
                l[i, j] = l[i, j] + x if lattice(i, j) else l[i, j] - x
            elif a[i, j] > 164:
                x = 31
                l[i, j] = l[i, j] + x if lattice(i, j) else l[i, j] - x
    l = l.clip(0, 255).astype(np.uint8)
    a = a.astype(np.uint8)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


def yellowandblue(im):
    w, h, _ = im.shape
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    b = b.astype(np.float64)
    a = a.astype(np.float64)
    for j in range(h):
        for i in range(w):
            if a[i, j] >= 133:
                x = 31 * (a[i, j] - 133) / (164 - 133)
                b[i, j] = b[i, j] + x if lattice(i, j) else b[i, j] - x
            elif a[i, j] > 164:
                x = 31
                b[i, j] = b[i, j] + x if lattice(i, j) else b[i, j] - x
    b = b.clip(0, 255).astype(np.uint8)
    a = a.astype(np.uint8)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)


# ======================
# Webアプリ化
# ======================
st.title("リアルタイム色覚補正カメラ 🎥")

filter_option = st.selectbox(
    "フィルタを選択してください",
    ["Original", "Deuteranope", "Dark", "Blue", "Yellow", "Dark & Bright", "Yellow & Blue"]
)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if filter_option == "Deuteranope":
            return Deuteranope(img)
        elif filter_option == "Dark":
            return dark(img)
        elif filter_option == "Blue":
            return blue(img)
        elif filter_option == "Yellow":
            return yellow(img)
        elif filter_option == "Dark & Bright":
            return darkandbright(img)
        elif filter_option == "Yellow & Blue":
            return yellowandblue(img)
        else:
            return img

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
