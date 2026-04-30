from __future__ import annotations

from typing import Any

from typing_extensions import Self


class GraphDiffusion:
    def __init__(self: Self, device: str) -> None: ...
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str | None = ...,
        strict: bool = ...,
        map_location: Any = ...,
    ) -> Self: ...
    def set_num_demos(self: Self, num_demos: int) -> None: ...
    def set_num_diffusion_steps(self: Self, num_diffusion_steps: int) -> None: ...
    def eval(self: Self) -> Self: ...
    def predict_actions(self: Self, full_sample: dict[str, Any]) -> tuple[Any, Any]: ...


def sample_to_cond_demo(
    sample: dict[str, Any],
    num_waypoints: int,
    num_points: int = ...,
) -> dict[str, Any]:
    ...
