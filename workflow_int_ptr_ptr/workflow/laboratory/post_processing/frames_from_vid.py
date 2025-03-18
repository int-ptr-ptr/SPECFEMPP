from pathlib import Path
from typing import Collection

import cv2
import numpy as np


def recover_frames(
    video: str | Path,
    frames: Collection[int],
    im_out: str | Path | None = None,
    nrows: int = -1,
    ncols: int = 1,
    rows_first: bool = True,
):
    num_captured_frames = len(frames)
    if nrows > 0 and ncols > 0:
        if nrows * ncols < num_captured_frames:
            raise ValueError(
                f"Cannot fit {num_captured_frames} into a {nrows}x{ncols} grid."
            )
    elif nrows <= 0 and ncols <= 0:
        if rows_first:
            nrows = 1
            ncols = num_captured_frames
        else:
            nrows = num_captured_frames
            ncols = 1
    elif nrows <= 0:
        # ncols > 0
        nrows = int(np.ceil(num_captured_frames / ncols))
    else:
        # ncols <= 0, nrows > 0
        ncols = int(np.ceil(num_captured_frames / nrows))

    cam = cv2.VideoCapture(str(video))
    framenum = 0
    ret, frame = cam.read()
    height, width, pix = frame.shape
    out = np.ones((height * nrows, width * ncols, pix), dtype=frame.dtype)

    saved_frames = 0
    while ret:
        # check if we want to save this frame
        if framenum in frames:
            if rows_first:
                x = saved_frames % ncols
                y = saved_frames // ncols
            else:
                y = saved_frames % nrows
                x = saved_frames // nrows

            out[y * height : (y + 1) * height, x * width : (x + 1) * width, :] = frame
            saved_frames += 1

        ret, frame = cam.read()
        framenum += 1

    if im_out is None:
        return out
    else:
        cv2.imwrite(str(im_out), out)
