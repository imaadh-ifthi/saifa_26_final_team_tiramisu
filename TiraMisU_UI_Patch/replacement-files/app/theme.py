"""Session-local native theme adapter; never mutate a shared palette."""
from collections.abc import Mapping
import streamlit as st

LIGHT = dict(navy="#302720", slate="#51483f", muted="#70665b", light_bg="#f3eee6",
 card_bg="#fffdf9", border="#d9d0c4", teal="#08786d", teal_light="#e0f0eb",
 blue="#315fb5", blue_light="#e6edf9", amber="#8c5b13", amber_light="#fff0d2",
 red="#b33a3a", red_light="#fbe8e5", green="#237044", green_light="#e4f1e5")
DARK = dict(navy="#f4eee5", slate="#ddd3c6", muted="#b6ab9c", light_bg="#292722",
 card_bg="#211f1b", border="#49443c", teal="#70d8c3", teal_light="#193d36",
 blue="#9ebdff", blue_light="#24344d", amber="#edc47c", amber_light="#42331e",
 red="#ffaaa3", red_light="#482b2a", green="#97d5a6", green_light="#253c2d")

def theme_name():
    return "dark" if st.context.theme.type == "dark" else "light"

class ThemePalette(Mapping):
    def __getitem__(self, key):
        return (DARK if theme_name() == "dark" else LIGHT)[key]
    def __iter__(self):
        return iter(LIGHT)
    def __len__(self):
        return len(LIGHT)

PALETTE = ThemePalette()
