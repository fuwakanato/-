#ver10/17.1
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="リアルタイム色覚補正", layout="centered")
st.title("🎥 リアルタイム色覚補正カメラ")
st.write("スマホやPCのブラウザで、リアルタイムにフィルタを切り替えます。")

# ========= フィルタ =========
def Deuteranope(im):
    b, g, r = cv2.split(im)
    bl = np.power((b/255+0.055)/1.055, 2.4)
    gl = np.power((g/255+0.055)/1.055, 2.4)
    rl = np.power((r/255+0.055)/1.055, 2.4)
    l = 0.31394*rl + 0.63957*gl + 0.04652*bl
    m = 0.15530*rl + 0.75796*gl + 0.08673*bl
    s = 0.01772*rl + 0.10945*gl + 0.87277*bl
    m = np.where(s <= l, 0.82781*l+0.17216*s, 0.81951*l+0.18046*s)
    rl = 5.47213*l - 4.64189*m + 0.16958*s
    gl = -1.12464*l + 2.29255*m - 0.16786*s
    bl = 0.02993*l - 0.19325*m + 1.16339*s
    bd = (np.power(bl,1/2.4)*1.055-0.055)*255
    gd = (np.power(gl,1/2.4)*1.055-0.055)*255
    rd = (np.power(rl,1/2.4)*1.055-0.055)*255
    return cv2.merge([
        bd.clip(0,255).astype(np.uint8),
        gd.clip(0,255).astype(np.uint8),
        rd.clip(0,255).astype(np.uint8)
    ])

def dark(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -128*(a-133)/(164-133), 0).clip(-128, 0)
    l = (l.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

def blue(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, -31*(a-133)/(164-133), 0).clip(-31, 0)
    b = (b.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

def yellow(im):
    imglab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(imglab)
    a = a.astype(np.float64)
    x = np.where(a >= 133, 31*(a-133)/(164-133), 0).clip(0, 31)
    b = (b.astype(np.float64) + x).clip(0, 255).astype(np.uint8)
    img = cv2.merge((l, a.astype(np.uint8), b))
    return cv2.cvtColor(img, cv2.COLOR_Lab2BGR)

FILTERS = {
    "Original": lambda x: x,
    "Deuteranope": Deuteranope,
    "Dark": dark,
    "Blue": blue,
    "Yellow": yellow,
}

selected_filter = st.selectbox("フィルタを選択してください", list(FILTERS.keys()))
resolution = st.select_slider("解像度（軽量化推奨）", ["320x240","640x480","960x540","1280x720"], value="640x480")
W, H = map(int, resolution.split("x"))
fps = st.slider("FPS", 5, 30, 15)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # 負荷軽減のため先にリサイズ
        img = cv2.resize(img, (W, H))
        return FILTERS[selected_filter](img)

# ========= STUN/TURN 設定 =========
# Secrets に [turn] を入れていれば TURN を使う（OpenRelay や Twilio 等）
if "turn" in st.secrets:
    ice_servers = [
        {"urls": st.secrets["turn"]["urls"]},
        {
            "urls": st.secrets["turn"]["urls"],
            "username": st.secrets["turn"]["username"],
            "credential": st.secrets["turn"]["credential"],
        },
    ]
else:
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]

# ========= WebRTC 起動（非同期＋一度だけ初期化） =========
webrtc_streamer(
    key="webrtc-camera",
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": ice_servers},
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": "environment"},  # スマホは背面カメラを優先
            "width": {"ideal": W},
            "height": {"ideal": H},
            "frameRate": {"ideal": fps},
        },
        "audio": False,
    },
    async_processing=True,   # 停止時の競合を減らす
)
st.caption("接続が不安定な場合は再読み込み、モバイル回線の切替、または別ブラウザでお試しください。")
