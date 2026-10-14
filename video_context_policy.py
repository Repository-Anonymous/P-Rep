"""World-model policy that supplies a rolling video context to V-JEPA."""

from collections import deque

import numpy as np
from stable_worldmodel.policy import WorldModelPolicy


class VideoContextWorldModelPolicy(WorldModelPolicy):
    """Seed each environment with a clip, then maintain a rolling frame window."""

    def __init__(self, *, initial_video_context, **kwargs):
        super().__init__(**kwargs)
        self.transform["pixels_video_context"] = self.transform["pixels"]
        clips = np.asarray(initial_video_context)
        if clips.ndim != 5:
            raise ValueError("initial_video_context must have shape (B,T,H,W,C)")
        self.context_len = clips.shape[1]
        self._video_context = [
            deque((frame.copy() for frame in clip), maxlen=self.context_len)
            for clip in clips
        ]
        self._first_observation = True

    def get_action(self, info_dict: dict, **kwargs):
        pixels = np.asarray(info_dict["pixels"])
        if not self._first_observation:
            for env_i, context in enumerate(self._video_context):
                context.append(pixels[env_i, -1].copy())
        self._first_observation = False

        video_info = dict(info_dict)
        # Keep the current single-frame observation in ``pixels`` so LeWM's
        # action horizon remains one. The adapter consumes this separate key.
        video_info["pixels_video_context"] = np.stack(
            [np.stack(context) for context in self._video_context]
        )
        return super().get_action(video_info, **kwargs)
