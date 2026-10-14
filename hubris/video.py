"""Small helpers around OpenCV video reading and writing."""
from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0


def video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def read_frames(path: str, frame_ids):
    """Yield (frame_id, image) for the requested frames, in ascending order.

    Seeks only when the gap is large; sequential reads are much faster than seeking.
    """
    cap = cv2.VideoCapture(path)
    pos = 0
    for fid in sorted(set(int(f) for f in frame_ids)):
        if fid - pos > 60 or fid < pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            pos = fid
        while pos < fid:
            cap.grab()
            pos += 1
        ok, img = cap.read()
        pos += 1
        if not ok:
            break
        yield fid, img
    cap.release()
