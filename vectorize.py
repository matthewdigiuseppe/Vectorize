#!/usr/bin/env python3
"""
vectorize.py — Convert raster images (PNG, JPEG, WEBP) to clean SVG vector files.

Usage examples:
    python vectorize.py input.png output.svg
    python vectorize.py input.jpg output.svg --mode color --colors 12
    python vectorize.py input.png output.svg --mode outline --threshold 128
    python vectorize.py --batch ./images --output-dir ./svgs --mode detailed
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Error: Pillow is required. Install it with: pip install Pillow")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_DIMENSION = 4000


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image, max_dim: int = MAX_DIMENSION) -> Image.Image:
    """Resize an image if either dimension exceeds *max_dim*, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img


def binarize_image(img: Image.Image, threshold: int = 128) -> Image.Image:
    """Convert an image to strict black/white using the given threshold."""
    gray = img.convert("L")
    return gray.point(lambda p: 255 if p > threshold else 0, mode="1")


def quantize_colors(img: Image.Image, n_colors: int) -> Image.Image:
    """Reduce an image to *n_colors* using median-cut quantization."""
    rgb = img.convert("RGB")
    return rgb.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")


# ---------------------------------------------------------------------------
# Vectorization back-ends
# ---------------------------------------------------------------------------

def _vtracer_available() -> bool:
    try:
        import vtracer  # noqa: F401
        return True
    except ImportError:
        return False


def _potrace_available() -> bool:
    return shutil.which("potrace") is not None


def vectorize_with_vtracer(
    img: Image.Image,
    mode: str,
    simplify: float,
    n_colors: int,
    threshold: int | None,
) -> str:
    """Produce SVG content via the *vtracer* library."""
    import vtracer

    # Prepare image bytes depending on mode
    if mode == "outline":
        processed = binarize_image(img, threshold if threshold is not None else 128)
        processed = processed.convert("RGBA")
    elif mode == "color":
        processed = quantize_colors(img, n_colors).convert("RGBA")
    else:  # detailed
        processed = img.convert("RGBA")

    buf = io.BytesIO()
    processed.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    # Map simplify (0–1) to vtracer parameters.
    # Higher simplify → fewer points, larger corner/segment thresholds.
    corner_threshold = int(30 + simplify * 120)  # 30–150
    segment_length = max(1.0, simplify * 10.0)
    splice_threshold = int(30 + simplify * 120)

    color_precision = {
        "outline": 1,
        "color": 6,
        "detailed": 8,
    }[mode]

    filter_speckle = {
        "outline": 4,
        "color": 4,
        "detailed": 2,
    }[mode]

    svg_str: str = vtracer.convert_raw_image_to_svg(
        raw_bytes,
        img_format="png",
        colormode="binary" if mode == "outline" else "color",
        mode="spline",
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=corner_threshold,
        length_threshold=segment_length,
        splice_threshold=splice_threshold,
    )
    return svg_str


def vectorize_with_potrace(
    img: Image.Image,
    mode: str,
    simplify: float,
    n_colors: int,
    threshold: int | None,
) -> str:
    """Produce SVG content by shelling out to *potrace*.

    potrace only handles monochrome input natively. For color/detailed modes we
    trace each quantized colour layer separately and merge the SVG paths.
    """
    if mode == "outline":
        return _potrace_single(img, simplify, threshold)

    # Color / detailed: quantize then trace each colour layer independently.
    n = n_colors if mode == "color" else min(n_colors * 2, 64)
    quantized = quantize_colors(img, n)
    w, h = quantized.size
    pixels = list(quantized.getdata())

    unique_colors = list(dict.fromkeys(pixels))

    paths_svg: list[str] = []
    for color in unique_colors:
        mask = Image.new("L", (w, h))
        mask_pixels = [255 if p == color else 0 for p in pixels]
        mask.putdata(mask_pixels)

        layer_svg = _potrace_trace_bmp(mask, simplify)
        if layer_svg is None:
            continue

        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        # Extract <path> elements and recolour them.
        for line in layer_svg.splitlines():
            stripped = line.strip()
            if stripped.startswith("<path"):
                coloured = stripped.replace('fill="#000000"', f'fill="{hex_color}"')
                coloured = coloured.replace("fill=\"#000000\"", f'fill="{hex_color}"')
                # If no fill attribute exists, inject one.
                if f'fill="{hex_color}"' not in coloured:
                    coloured = coloured.replace("<path", f'<path fill="{hex_color}"', 1)
                paths_svg.append(coloured)

    svg_header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
    )
    svg_footer = "</svg>\n"
    return svg_header + "\n".join(paths_svg) + "\n" + svg_footer


def _potrace_single(img: Image.Image, simplify: float, threshold: int | None) -> str:
    """Trace a single monochrome image with potrace."""
    bw = binarize_image(img, threshold if threshold is not None else 128)
    result = _potrace_trace_bmp(bw.convert("L"), simplify)
    if result is None:
        raise RuntimeError("potrace produced no output")
    return result


def _potrace_trace_bmp(gray_img: Image.Image, simplify: float) -> str | None:
    """Write a PGM, call potrace, return SVG string or None."""
    with tempfile.NamedTemporaryFile(suffix=".pgm", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_out:
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name

    try:
        gray_img.save(tmp_in_path)

        # potrace tolerances: -O is optimisation tolerance, -t is threshold.
        opt_tolerance = simplify * 5.0  # 0–5
        cmd = [
            "potrace",
            tmp_in_path,
            "-b", "svg",
            "-O", str(opt_tolerance),
            "-o", tmp_out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        return Path(tmp_out_path).read_text()
    except subprocess.CalledProcessError as exc:
        print(f"Warning: potrace failed: {exc.stderr.decode(errors='replace')}", file=sys.stderr)
        return None
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------

def vectorize_pil_image(
    img: Image.Image,
    mode: str = "outline",
    n_colors: int = 8,
    simplify: float = 0.3,
    threshold: int | None = None,
) -> str:
    """Vectorize an in-memory PIL Image and return the SVG string.

    This is the core conversion entry point used by both the CLI and the web
    dashboard.
    """
    img = preprocess_image(img)

    if _vtracer_available():
        return vectorize_with_vtracer(img, mode, simplify, n_colors, threshold)
    elif _potrace_available():
        print("Note: vtracer not found, falling back to potrace.", file=sys.stderr)
        return vectorize_with_potrace(img, mode, simplify, n_colors, threshold)
    else:
        raise RuntimeError(
            "No vectorization backend available. "
            "Install vtracer (pip install vtracer) or potrace (apt install potrace)."
        )


def convert_image(
    input_path: str,
    output_path: str,
    mode: str = "outline",
    n_colors: int = 8,
    simplify: float = 0.3,
    threshold: int | None = None,
    preview: bool = False,
) -> None:
    """Load a raster image, vectorize it, and write the SVG to *output_path*."""
    in_path = Path(input_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = in_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    img = Image.open(in_path)
    svg_content = vectorize_pil_image(img, mode, n_colors, simplify, threshold)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg_content)
    print(f"Saved: {out}")

    if preview:
        webbrowser.open(out.resolve().as_uri())


def process_batch(
    input_dir: str,
    output_dir: str,
    mode: str,
    n_colors: int,
    simplify: float,
    threshold: int | None,
    preview: bool,
) -> None:
    """Vectorize every supported image in *input_dir*, writing SVGs to *output_dir*."""
    in_dir = Path(input_dir)
    if not in_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        print(f"No supported images found in {in_dir}", file=sys.stderr)
        return

    successes = 0
    failures = 0
    for f in files:
        svg_name = f.stem + ".svg"
        out_path = out_dir / svg_name
        try:
            convert_image(
                str(f), str(out_path), mode, n_colors, simplify, threshold, preview=False
            )
            successes += 1
        except Exception as exc:
            print(f"Error processing {f.name}: {exc}", file=sys.stderr)
            failures += 1

    print(f"\nBatch complete: {successes} succeeded, {failures} failed.")

    if preview and successes > 0:
        # Open the first generated SVG for a quick check.
        first_svg = sorted(out_dir.glob("*.svg"))[0]
        webbrowser.open(first_svg.resolve().as_uri())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raster images (PNG, JPEG, WEBP) to clean SVG vector files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s logo.png logo.svg\n"
            "  %(prog)s photo.jpg out.svg --mode color --colors 16\n"
            "  %(prog)s icon.png icon.svg --mode outline --threshold 100\n"
            "  %(prog)s --batch ./images --output-dir ./svgs --mode detailed\n"
        ),
    )

    parser.add_argument("input", nargs="?", help="Input image path (ignored when --batch is used)")
    parser.add_argument("output", nargs="?", help="Output SVG path (ignored when --batch is used)")

    parser.add_argument(
        "--mode",
        choices=["outline", "color", "detailed"],
        default="outline",
        help="Vectorization mode (default: outline)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=8,
        metavar="N",
        help="Number of colours for color mode (default: 8)",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.3,
        metavar="FLOAT",
        help="Path simplification tolerance, 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        metavar="INT",
        help="Manual binarization threshold for outline mode (0-255)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open the resulting SVG in the default browser",
    )

    # Batch mode
    parser.add_argument(
        "--batch",
        metavar="DIR",
        help="Process all supported images in DIR",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Output directory for batch mode (required with --batch)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate simplify range.
    if not 0.0 <= args.simplify <= 1.0:
        parser.error("--simplify must be between 0 and 1")

    # Validate threshold range.
    if args.threshold is not None and not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")

    # Validate colors.
    if args.colors < 2:
        parser.error("--colors must be at least 2")

    if args.batch:
        if not args.output_dir:
            parser.error("--output-dir is required when using --batch")
        try:
            process_batch(
                args.batch,
                args.output_dir,
                args.mode,
                args.colors,
                args.simplify,
                args.threshold,
                args.preview,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.input or not args.output:
            parser.error("input and output are required (or use --batch)")
        try:
            convert_image(
                args.input,
                args.output,
                args.mode,
                args.colors,
                args.simplify,
                args.threshold,
                args.preview,
            )
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
