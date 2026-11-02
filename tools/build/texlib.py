"""Small image helpers used by the asset build: background removal for the
flat-lay texture sheets, colour dilation (edge padding) and normal-map
generation from albedo detail."""
import numpy as np
from PIL import Image
from scipy import ndimage


def load_rgb(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255.0


def save_rgb(path, arr, quality=92):
    img = Image.fromarray((np.clip(arr[..., :3], 0, 1) * 255 + 0.5).astype(np.uint8))
    if path.endswith('.jpg'):
        img.save(path, quality=quality, subsampling=0)
    else:
        img.save(path, optimize=True)


def save_rgba(path, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGBA').save(path, optimize=True)


def cut_piece(sheet, box, tol=0.075, pad=6, erode=1):
    """Crop `box` (x0,y0,x1,y1) from the sheet and return RGBA where the
    flat grey backdrop (flood-filled from the crop border) is transparent."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(sheet.shape[1], x1 + pad), min(sheet.shape[0], y1 + pad)
    crop = sheet[y0:y1, x0:x1].copy()
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    bg = np.median(border, axis=0)
    # backdrop = low-saturation pixels close to the backdrop colour, connected to the border
    dist = np.abs(crop - bg).max(axis=2)
    sat = crop.max(2) - crop.min(2)
    cand = (dist < tol) & (sat < 0.06)
    lab, _ = ndimage.label(cand)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgmask = np.isin(lab, edge_labels[edge_labels > 0])
    fg = ~bgmask
    fg = ndimage.binary_opening(fg, iterations=1)
    fg = ndimage.binary_fill_holes(fg)
    # keep the largest component only (drops neighbouring pieces' slivers)
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    if erode:
        fg = ndimage.binary_erosion(fg, iterations=erode)  # drop anti-aliased backdrop fringe
    alpha = fg.astype(np.float32)
    return np.dstack([crop, alpha]), (x0, y0)


def dilate_colors(rgba, iters=48):
    """Push foreground colours outward into transparent pixels (edge padding)
    so bilinear/mip sampling near UV borders never picks up backdrop grey."""
    rgb = rgba[..., :3].copy()
    known = rgba[..., 3] > 0.5
    rgb[~known] = 0
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32)
    for _ in range(iters):
        if known.all():
            break
        cnt = ndimage.convolve(known.astype(np.float32), k, mode='nearest')
        acc = np.stack([ndimage.convolve(rgb[..., c] * known, k, mode='nearest') for c in range(3)], -1)
        grow = (~known) & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow][:, None]
        known = known | grow
    if not known.all():
        mean = rgb[known].mean(axis=0)
        rgb[~known] = mean
    return rgb


def luminance(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def normal_from_albedo(rgb, strength=2.0, detail_sigma=6.0, mask=None):
    """Tangent-space normal map (OpenGL/glTF convention, +Y up) from the
    high-frequency part of the albedo luminance."""
    lum = luminance(rgb)
    h = lum - ndimage.gaussian_filter(lum, detail_sigma)
    h = ndimage.gaussian_filter(h, 0.7)
    if mask is not None:
        h = h * mask
    dx = ndimage.sobel(h, axis=1) * strength
    dy = ndimage.sobel(h, axis=0) * strength
    n = np.dstack([-dx, dy, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return n * 0.5 + 0.5


def resample_rows(img, src_keys, dst_keys, out_h):
    """Piecewise-linear vertical remap: output row r takes source row
    interp(r, dst_keys, src_keys)."""
    rows = np.interp(np.arange(out_h) + 0.5, dst_keys, src_keys)
    coords = np.stack(np.meshgrid(rows - 0.5, np.arange(img.shape[1]), indexing='ij'))
    return np.dstack([ndimage.map_coordinates(img[..., c], coords, order=1, mode='nearest') for c in range(img.shape[2])])


def extend_band(img, y0, y1, extra):
    """Insert `extra` rows after row y1 by mirror-repeating band [y0, y1)."""
    band = img[y0:y1]
    reps = []
    flip = True
    total = 0
    while total < extra:
        b = band[::-1] if flip else band
        reps.append(b)
        total += len(b)
        flip = not flip
    ins = np.concatenate(reps)[:extra]
    return np.concatenate([img[:y1], ins, img[y1:]])
