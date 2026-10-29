"""Batch 4x upscale from the command line.

    python cli.py                    # every image in ./input
    python cli.py a.png b.jpg --dpi 300 --tile 512
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from upscaler import Upscaler

HERE = Path(__file__).resolve().parent
EXTS = (".png", ".jpg", ".jpeg", ".webp")


def main():
    ap = argparse.ArgumentParser(description="4x AI upscaler (Real-ESRGAN)")
    ap.add_argument("files", nargs="*", help="images to upscale (default: all in ./input)")
    ap.add_argument("-o", "--out", default=str(HERE / "output"), help="output folder")
    ap.add_argument("--tile", type=int, default=512, help="tile size; lower it if you run out of VRAM")
    ap.add_argument("--dpi", type=int, default=300, help="DPI tag written into the output")
    args = ap.parse_args()

    in_dir = HERE / "input"
    files = [Path(f) for f in args.files] or (
        sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS) if in_dir.is_dir() else [])
    if not files:
        sys.exit("No images found. Pass files or put PNG/JPG/WEBP images in ./input")

    up = Upscaler()
    print(f"Device: {up.device_name}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        print(f"Upscaling {f.name} ...")
        result = up.upscale(Image.open(f), args.tile)
        dst = out_dir / f"{f.stem}_x4.png"
        result.save(dst, dpi=(args.dpi, args.dpi))
        w, h = result.size
        print(f"  -> {dst.name}  {w}x{h}px  ({w / args.dpi * 2.54:.1f} x {h / args.dpi * 2.54:.1f} cm at {args.dpi} DPI)")
    print("Done.")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="backslashreplace")
    main()
