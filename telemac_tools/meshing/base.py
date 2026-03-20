"""Abstract mesh generation interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from telemac_tools.model import TelemacDomain, Mesh2D


class MeshBackend(ABC):
    """Base class for mesh generation backends."""

    @abstractmethod
    def generate(self, domain: TelemacDomain, min_angle: float = 30.0,
                 max_area: float | None = None) -> Mesh2D:
        """Generate a triangular mesh for the given domain."""
        ...
