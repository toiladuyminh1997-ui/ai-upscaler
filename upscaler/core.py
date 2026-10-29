"""Real-ESRGAN (x4plus) 4x upscaler core: model definition, loading and tiled inference."""
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "RealESRGAN_x4plus.pth"
SCALE = 4


# --- RRDBNet (the Real-ESRGAN generator architecture) ---------------------------------

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# --- Upscaler -------------------------------------------------------------------------

class Upscaler:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if not MODEL_PATH.exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading model weights to {MODEL_PATH} ...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        self.model = RRDBNet()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device, self.dtype)

    @property
    def device_name(self):
        return torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU (slow)"

    @torch.inference_mode()
    def _tiled(self, img, tile, pad=16):
        """img: 1x3xHxW tensor in [0,1]. Returns 1x3x(4H)x(4W)."""
        _, _, h, w = img.shape
        out = torch.empty((1, 3, h * SCALE, w * SCALE), dtype=img.dtype, device=img.device)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                # tile with padding so seams get context from neighbours
                y0, y1 = max(y - pad, 0), min(y + tile + pad, h)
                x0, x1 = max(x - pad, 0), min(x + tile + pad, w)
                sr = self.model(img[:, :, y0:y1, x0:x1])
                ty, tx = (y - y0) * SCALE, (x - x0) * SCALE
                th, tw = (min(y + tile, h) - y) * SCALE, (min(x + tile, w) - x) * SCALE
                out[:, :, y * SCALE:y * SCALE + th, x * SCALE:x * SCALE + tw] = sr[:, :, ty:ty + th, tx:tx + tw]
        return out

    def upscale(self, im: Image.Image, tile=512) -> Image.Image:
        alpha = None
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            alpha = im.getchannel("A")
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        sr = self._tiled(t, tile).clamp(0, 1)
        out = (sr.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
        result = Image.fromarray(out, "RGB")
        if alpha is not None:  # alpha channel: a plain high-quality resize is fine
            result.putalpha(alpha.resize(result.size, Image.LANCZOS))
        return result
