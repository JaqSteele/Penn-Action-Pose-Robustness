"""Penn Action frame files: naming, listing and D-01 verification.

Frames are stored outside the repository as ``<root>/frames/<video_id>/<NNNNNN>.jpg``, numbered
from 000001 to ``nframes`` with no gaps. Verification compares each video folder with its
annotation file (frame count, image size after the documented 481/482 -> 480 width correction)
and checks that every JPEG can be read. Nothing is repaired or modified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from .labels import DEFAULT_DATA_ROOT, PennVideo

PathLike = Union[str, Path]
FRAME_NAME = re.compile(r"^(\d{6})\.jpg$")


def frames_root(root: PathLike = DEFAULT_DATA_ROOT) -> Path:
    return Path(root) / "frames"


def video_frame_dir(video_id: str, root: PathLike = DEFAULT_DATA_ROOT) -> Path:
    return frames_root(root) / video_id


def frame_name(index_1based: int) -> str:
    return f"{index_1based:06d}.jpg"


def frame_paths(video_id: str, n_frames: int, root: PathLike = DEFAULT_DATA_ROOT) -> List[Path]:
    """Expected frame paths 000001.jpg ... in order (existence is not checked here)."""
    directory = video_frame_dir(video_id, root)
    return [directory / frame_name(i) for i in range(1, n_frames + 1)]


@dataclass
class FrameCheck:
    video_id: str
    n_expected: int
    n_found: int = 0
    issues: List[str] = field(default_factory=list)
    image_size: Optional[tuple] = None       # (width, height) of the frames, if consistent

    @property
    def ok(self) -> bool:
        return not self.issues


def verify_video_frames(video: PennVideo, root: PathLike = DEFAULT_DATA_ROOT,
                        check_images: bool = True) -> FrameCheck:
    """D-01 check of one video's frame folder against its annotations."""
    result = FrameCheck(video_id=video.video_id, n_expected=video.n_frames)
    directory = video_frame_dir(video.video_id, root)
    if not directory.is_dir():
        result.issues.append("frame folder missing")
        return result
    entries = sorted(p.name for p in directory.iterdir())
    names = [n for n in entries if FRAME_NAME.match(n)]
    others = [n for n in entries if not FRAME_NAME.match(n)]
    result.n_found = len(names)
    if others:
        result.issues.append(f"unexpected files: {others[:5]}{' ...' if len(others) > 5 else ''}")
    expected = [frame_name(i) for i in range(1, video.n_frames + 1)]
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        if missing:
            result.issues.append(f"{len(missing)} missing frames (first: {missing[:3]})")
        if extra:
            result.issues.append(f"{len(extra)} frames beyond nframes (first: {extra[:3]})")
    if len(names) != video.n_frames:
        result.issues.append(f"frame count {len(names)} != nframes {video.n_frames}")
    if check_images and names:
        from PIL import Image  # imported lazily: only needed for image checks
        sizes = set()
        unreadable = []
        for n in names:
            try:
                with Image.open(directory / n) as im:
                    size = im.size
                    fmt = im.format
                    im.verify()
                if fmt != "JPEG":
                    unreadable.append(f"{n} (format {fmt})")
                sizes.add(size)
            except Exception as exc:  # noqa: BLE001 - any read failure is a reportable issue
                unreadable.append(f"{n} ({type(exc).__name__})")
        if unreadable:
            result.issues.append(f"{len(unreadable)} unreadable frames (first: {unreadable[:3]})")
        if len(sizes) == 1:
            result.image_size = next(iter(sizes))
            expected_size = (video.image_width, video.image_height)
            if result.image_size != expected_size:
                result.issues.append(f"image size {result.image_size} != annotation "
                                     f"(corrected) {expected_size}")
        elif len(sizes) > 1:
            result.issues.append(f"inconsistent frame sizes: {sorted(sizes)[:4]}")
    return result
