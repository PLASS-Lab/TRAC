"""Joint bilateral filtering as a scalar-query softmax readout, with a two-read cache.

For pixel p, O(p) = sum_j G(p-j) exp(-(g_p-g_j)^2 / 2 sr^2) D(j) / sum_j (same).
Dropping the factor exp(-g_p^2 / 2 sr^2) that does not depend on j, this is
softmax_j(t_p * kappa_j + b_j) D(j) with t_p = g_p / sr^2 in [0, 1/sr^2],
kappa_j = g_j in [0, 1] and positive weights b_j = log G(p-j) - g_j^2 / 2 sr^2:
the family of the interpolation lemma with K = 1 and query half-width
T = 1 / (2 sr^2). Storing the filtered image J_s at S guide levels (the cache,
S Gaussian filters) and reading two neighbouring levels per pixel is the
Durand-Dorsey / Yang et al. fast bilateral scheme, with error at most
R / (16 sr^4 (S-1)^2) for |D| <= R.
"""
import torch
import torch.nn.functional as F


def gaussian_kernel(sigma, device, dtype=torch.float32):
    radius = max(1, int(3 * sigma + .5))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum(), radius


def blur(x, sigma):
    """Separable Gaussian blur of (B, C, H, W) with replicate padding."""
    k, r = gaussian_kernel(sigma, x.device, x.dtype)
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode='replicate')
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def error_bound(sigma_r, snapshots, value_range=1.0):
    return value_range / (16 * sigma_r ** 4 * (snapshots - 1) ** 2)


def _stable_snapshot(depth, guide, level, sigma_s, sigma_r):
    """Normalized snapshot, with a log-weight fallback for underflowed windows.

    A common weight rescaling cancels in the ratio. Unlike a denominator
    floor it preserves the linear, constant-preserving mathematical operator.
    The fallback uses the same finite window and replicated boundaries as blur.
    """
    logk = -(level - guide).square() / (2 * sigma_r ** 2)
    logk = logk - logk.amax(dim=(-2, -1), keepdim=True)
    k = logk.exp()
    sums = blur(torch.cat((k * depth, k), 1), sigma_s)
    numerator, denominator = sums[:, :1], sums[:, 1:]
    safe = denominator >= 32 * torch.finfo(denominator.dtype).tiny
    result = numerator / torch.where(safe, denominator, torch.ones_like(denominator))
    bad = (~safe[:, 0]).nonzero(as_tuple=False)
    if bad.numel():
        kernel, radius = gaussian_kernel(sigma_s, depth.device, depth.dtype)
        spatial = (kernel[:, None] * kernel[None, :]).flatten().log()
        padded_g = F.pad(guide, (radius,) * 4, mode='replicate')[:, 0]
        padded_v = F.pad(depth, (radius,) * 4, mode='replicate')[:, 0]
        oy, ox = torch.meshgrid(
            torch.arange(2 * radius + 1, device=depth.device),
            torch.arange(2 * radius + 1, device=depth.device), indexing='ij')
        oy, ox = oy.flatten(), ox.flatten()
        for indices in bad.split(1024):
            batch, yy, xx = indices.unbind(1)
            local_g = padded_g[batch[:, None], yy[:, None] + oy, xx[:, None] + ox]
            local_v = padded_v[batch[:, None], yy[:, None] + oy, xx[:, None] + ox]
            logits = spatial - (level - local_g).square() / (2 * sigma_r ** 2)
            result[batch, 0, yy, xx] = (logits.softmax(-1) * local_v).sum(-1)
    return result[:, 0]


def bilateral_two_read(depth, guide, sigma_s, sigma_r, snapshots, levels=None):
    """depth (B,1,H,W) any range, guide (B,1,H,W) in [0,1]. Cache of S filtered images.

    levels: optional increasing guide levels (S,) spanning [0,1]; default uniform, which
    is the case the error bound covers."""
    if snapshots < 2 or sigma_r <= 0 or sigma_s <= 0:
        raise ValueError('snapshots >= 2 and positive bandwidths are required')
    original_dtype = depth.dtype
    if depth.dtype in (torch.float16, torch.bfloat16):
        depth, guide = depth.float(), guide.float()
    if levels is None:
        levels = torch.linspace(0, 1, snapshots, device=depth.device, dtype=depth.dtype)
    else:
        levels = torch.as_tensor(levels, device=depth.device, dtype=depth.dtype)
    if levels.ndim != 1 or len(levels) != snapshots or not bool((levels[1:] > levels[:-1]).all()):
        raise ValueError('levels must contain snapshots strictly increasing entries')
    if bool((guide < levels[0]).any() or (guide > levels[-1]).any()):
        raise ValueError('levels must cover the guide range')
    cache = torch.stack([_stable_snapshot(depth, guide, g, sigma_s, sigma_r)
                         for g in levels], 1)         # (B, S, H, W)
    g = guide.clamp(0, 1)[:, 0]
    low = (torch.searchsorted(levels, g.contiguous(), right=True) - 1).clamp(0, snapshots - 2)
    frac = ((g - levels[low]) / (levels[low + 1] - levels[low])).clamp(0, 1).unsqueeze(1)
    lo = cache.gather(1, low.unsqueeze(1))
    hi = cache.gather(1, (low + 1).unsqueeze(1))
    return (lo + frac * (hi - lo)).to(original_dtype)


def bilateral_exact(depth, guide, sigma_s, sigma_r, chunk=64):
    """Reference: direct truncated Gaussian-window sum (radius 3 sigma_s)."""
    _, r = gaussian_kernel(sigma_s, depth.device)
    b, _, h, w = depth.shape
    d = F.pad(depth, (r, r, r, r), mode='replicate')
    g = F.pad(guide, (r, r, r, r), mode='replicate')
    offsets = torch.arange(-r, r + 1, device=depth.device, dtype=depth.dtype)
    spatial = torch.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * sigma_s ** 2))
    out = torch.empty_like(depth)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        dw = F.unfold(d[:, :, y0:y1 + 2 * r], 2 * r + 1)          # (B, K, L)
        gw = F.unfold(g[:, :, y0:y1 + 2 * r], 2 * r + 1)
        gc = guide[:, :, y0:y1].reshape(b, 1, -1)
        wgt = spatial.reshape(1, -1, 1) * torch.exp(-(gw - gc) ** 2 / (2 * sigma_r ** 2))
        out[:, :, y0:y1] = ((wgt * dw).sum(1) / wgt.sum(1)).reshape(b, 1, y1 - y0, w)
    return out


def demo():
    torch.manual_seed(0)
    guide = torch.rand(1, 1, 48, 48).cumsum(-1).cumsum(-2)
    guide = (guide - guide.min()) / (guide.max() - guide.min())
    depth = torch.rand(1, 1, 48, 48)
    for sr, s in ((0.3, 8), (0.3, 32), (0.15, 32)):
        e = bilateral_exact(depth, guide, 2.0, sr)
        a = bilateral_two_read(depth, guide, 2.0, sr, s)
        err, bound = float((e - a).abs().max()), error_bound(sr, s)
        assert err <= bound, (sr, s, err, bound)
        print(sr, s, err, bound)


if __name__ == '__main__':
    demo()
