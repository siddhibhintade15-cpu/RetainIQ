"""Tiny page registry so any page can jump to another one."""

import streamlit as st

PAGES = {}


def register(name, page):
    PAGES[name] = page


def go_to(name):
    st.switch_page(PAGES[name])
