"""
Author: Core447
Year: 2024

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This programm comes with ABSOLUTELY NO WARRANTY!

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

from functools import lru_cache
import hashlib
import os
import sys
import threading
import time
from PIL import Image, ImageOps
import cv2
from loguru import logger as log
import globals as gl

VID_CACHE = os.path.join(gl.DATA_PATH, "cache", "videos")

class VideoFrameCache:
    def __init__(self, video_path, size: tuple[int, int]):
        self.lock = threading.Lock()

        self.video_path = video_path
        self.size = size
        self.cache = {}
        self.last_decoded_frame = None
        self.last_frame_index = -1

        # Per-frame delays in ms, only populated for GIFs
        self.frame_delays: list[int] = []

        if self._is_gif():
            self._gif_image = Image.open(video_path)
            self.n_frames = getattr(self._gif_image, "n_frames", 1)
            self._read_gif_delays()
            self.cap = None
        else:
            self._gif_image = None
            self.cap = cv2.VideoCapture(video_path)
            self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_md5 = self.get_video_hash()

        self.load_cache()

        self.do_caching = gl.settings_manager.get_app_settings().get("performance", {}).get("cache-videos", True)
        self.do_caching = True

        # Pre-cache all GIF frames upfront so playback never blocks
        if self._is_gif() and not self.is_cache_complete():
            for i in range(self.n_frames):
                self._get_gif_frame(i)

        if self.is_cache_complete():
            log.info("Cache is complete. Closing the video capture.")
            self.release()
        else:
            log.info("Cache is not complete. Continuing with video capture.")

        # Print size of cache in memory in mb:
        log.trace(f"Size of cache in memory: {sys.getsizeof(self.cache) / 1024 / 1024:.2f} MB")

        if not self._is_gif():
            log.trace(f"Size of capture: {sys.getsizeof(self.cap) / 1024 / 1024:.2f} MB")

    def _is_gif(self) -> bool:
        return os.path.splitext(self.video_path)[1].lower() == ".gif"

    def _read_gif_delays(self):
        """Read per-frame delays (ms) from GIF metadata."""
        try:
            for i in range(self.n_frames):
                self._gif_image.seek(i)
                delay = self._gif_image.info.get("duration", 0)
                self.frame_delays.append(delay)
            # Replace zero delays with the previous frame's delay (or 100ms fallback).
            # A 0ms delay is common on the last frame as an end-marker.
            last_valid = 100
            for i, d in enumerate(self.frame_delays):
                if d <= 0:
                    self.frame_delays[i] = last_valid
                else:
                    last_valid = d
        except Exception:
            self.frame_delays = [100] * self.n_frames
        finally:
            # Re-open the image so internal frame state is clean for playback.
            # Pillow GIF frames depend on previous frames (disposal method), so
            # seeking through all frames during delay-reading corrupts the state.
            try:
                self._gif_image.close()
            except Exception:
                pass
            self._gif_image = Image.open(self.video_path)

    def get_frame_delay(self, frame_index: int) -> int:
        """Return the delay in ms for the given GIF frame. Falls back to 100ms."""
        if self.frame_delays and 0 <= frame_index < len(self.frame_delays):
            return self.frame_delays[frame_index]
        return 100

    def _get_gif_frame(self, n: int) -> Image.Image:
        """Read frame n from a GIF via PIL, preserving RGBA transparency."""
        n = min(n, self.n_frames - 1)
        if n in self.cache:
            return self.cache[n]
        with self.lock:
            self._gif_image.seek(n)
            frame = self._gif_image.convert("RGBA")
        frame = ImageOps.fit(frame, self.size, Image.Resampling.LANCZOS)
        if self.do_caching:
            self.cache[n] = frame
            self.write_cache(frame, n)
        return frame

    def get_frame(self, n):
        if self._is_gif():
            return self._get_gif_frame(n)

        n = min(n, self.n_frames - 1)
        if self.is_cache_complete():
            return self.cache.get(n, None)

        # Otherwise, continue with video capture
        # Check if the frame is already decoded
        if n in self.cache:
            return self.cache[n]

        # If the requested frame is before the last decoded one, reset the capture
        if n < self.last_frame_index:
            with self.lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            self.last_frame_index = n - 1

        # Decode frames until the nth frame
        while self.last_frame_index < n:
            with self.lock:
                success, frame = self.cap.read()
            if not success:
                break  # Reached the end of the video
            self.last_frame_index += 1

            # Calculate the new height to maintain aspect ratio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            # Fill a 72x72 square completely with the image, keeping the aspect ratio
            pil_image = ImageOps.fit(pil_image, self.size, Image.Resampling.LANCZOS)

            self.last_decoded_frame = pil_image
            if self.do_caching:
                self.cache[self.last_frame_index] = pil_image

                # Write the frame to the cache
                self.write_cache(pil_image, self.last_frame_index)

        # Return the last decoded frame if the nth frame is not available
        return self.cache.get(n, self.last_decoded_frame)

    def release(self):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            if self._gif_image is not None:
                try:
                    self._gif_image.close()
                except Exception:
                    pass

    def get_video_hash(self) -> str:
        with self.lock:
            sha1sum = hashlib.md5()
            with open(self.video_path, 'rb') as video:
                block = video.read(2**16)
                while len(block) != 0:
                    sha1sum.update(block)
                    block = video.read(2**16)
                return sha1sum.hexdigest()

    def write_cache(self, image: Image, frame_index: int, key_index: int = None):
        """
        key_index: if None: single key video, if int: key index
        """
        # GIFs are cached as PNG to preserve the alpha channel
        ext = "png" if self._is_gif() else "jpg"
        with self.lock:
            if key_index is None:
                path = os.path.join(VID_CACHE, "single_key", self.video_md5, f"{self.size[0]}x{self.size[1]}", f"{frame_index}.{ext}")
            else:
                path = os.path.join(VID_CACHE, f"key: {key_index}", self.video_md5, f"{self.size[0]}x{self.size[1]}", f"{key_index}", f"{frame_index}.{ext}")

            if os.path.isfile(path):
                return

            os.makedirs(os.path.dirname(path), exist_ok=True)

            image.save(path)

    def load_cache(self, key_index: int = None):
        # GIFs are cached as PNG to preserve the alpha channel
        ext = "png" if self._is_gif() else "jpg"
        with self.lock:
            start = time.time()
            if key_index is None:
                path = os.path.join(VID_CACHE, "single_key", self.video_md5, f"{self.size[0]}x{self.size[1]}")
                if not os.path.exists(path):
                    return
                for file in os.listdir(path):
                    if os.path.splitext(file)[1] != f".{ext}":
                        continue
                    with Image.open(os.path.join(path, file)) as img:
                        self.cache[int(file.split(".")[0])] = img.copy()

            else:
                path = os.path.join(VID_CACHE, f"key: {key_index}", self.video_md5, f"{self.size[0]}x{self.size[1]}", f"{key_index}")
                if not os.path.exists(path):
                    return
                for file in os.listdir(path):
                    if os.path.splitext(file)[1] != f".{ext}":
                        continue
                    with Image.open(os.path.join(path, file)) as img:
                        self.cache[int(file.split(".")[0])] = img.copy()

            log.info(f"Loaded cache in {time.time() - start:.2f} seconds")

    @lru_cache(maxsize=None)
    def is_cache_complete(self) -> bool:
        return len(self.cache) == self.n_frames
