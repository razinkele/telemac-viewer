"""Pure functions used by `app.update_map` to decide full vs partial update.

Kept in a separate module so they can be unit-tested without booting
the full Shiny app (which requires a session, reactive scope, etc.).
"""

from __future__ import annotations


def decide_dispatch(*, prev_sig: tuple | None, curr_sig: tuple) -> str:
    """Return "full" if the structural signature changed, else "partial".

    A "full" update resends every layer, widget, view-state and effect.
    A "partial" update sends only the mesh color patch plus widget replacement.
    """
    if prev_sig is None:
        return "full"
    return "partial" if prev_sig == curr_sig else "full"
