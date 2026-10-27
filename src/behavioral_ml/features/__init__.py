"""Reusable feature assembly shared by notebooks and command-line experiments."""

from . import key, mouse


def build_feature_sets(mouse_events, key_events):
    """Return the 18-key, 21-mouse, and combined attempt-level feature tables."""
    presses = key.add_key_dynamics(key_events)
    keys = (
        key.aggregate_basic_key_dynamics(presses)
        .join(key.aggregate_selected_key_counts(presses, {
            "numpad": key.NUMPAD_KEYS, "tabs": key.TAB_KEYS,
            "correction": key.CORRECTION_KEYS, "enter": key.ENTER_KEYS,
        }))
        .join(key.aggregate_selected_key_dynamics(
            presses, {"shifts": key.SHIFT_KEYS}, ["Dwell Time", "Flight Time", "UU"]))
    )
    movements = mouse.aggregate_movement_features(mouse.add_mouse_kinematics(mouse_events))
    buttons = mouse.aggregate_button_features(mouse.add_button_features(mouse_events))
    mice = movements.join(buttons, how="outer").reindex(keys.index)
    return {"keys": keys, "mouse": mice, "keys_mouse": keys.join(mice)}
