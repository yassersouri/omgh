import numpy as np
from parts import *


def thresh_segment(seg, thresh):
    return seg >= thresh


def thresh_segment_max(seg):
    return thresh_segment(seg, np.max(seg))


def thresh_segment_mean(seg):
    return thresh_segment(seg, np.mean(seg))


def gen_part_points(part_rect, seg, parts):
    N = 100
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=xmin, high=xmax, size=N)
    ys = np.random.uniform(low=ymin, high=ymax, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if seg[x, y, 0]:
            parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
    print len(parts)
    return parts


def gen_bg_points(part_rect, seg, parts):
    N = 500
    h, w = seg.shape[0], seg.shape[1]
    xmin, xmax, ymin, ymax = part_rect

    xs = np.random.uniform(low=0, high=h-1, size=N)
    ys = np.random.uniform(low=0, high=w-1, size=N)

    parts = Parts()

    for x, y in zip(xs, ys):
        if (xmin <= x <= xmax and ymin <= y <= ymax):
            if not seg[x-2:x+3, y-2:y+3, 0].sum() and False:  # is this really necessary!?
                parts.append(Part(-1, '?', -1, int(round(y)), int(round(x)), 1))
        else:
            parts.append(Part(-1, '?', -1, int((round(y))), int(round(x)), 1))

    print len(parts)
    return parts
