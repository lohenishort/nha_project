#!/usr/bin/env python3
"""
Data loading utilities for LRA (Long Range Arena) and SCROLLS benchmarks.
"""

from datasets import load_dataset, Dataset

def load_pathfinder(variant: str = "curv_contour_length", split: str = "train") -> Dataset:
    """
    Loads the LRA Pathfinder dataset variant.

    Args:
        variant: The variant to load (default: "curv_contour_length").
        split: The dataset split (default: "train").

    Returns:
        Hugging Face Dataset instance.
    """
    return load_dataset("LRA/pathfinder", variant, split=split)

def load_listops(split: str = "train") -> Dataset:
    """
    Loads the LRA ListOps dataset.

    Args:
        split: The dataset split (default: "train").

    Returns:
        Hugging Face Dataset instance.
    """
    return load_dataset("LRA/listops", split=split)

def load_scrolls_qasper(split: str = "train") -> Dataset:
    """
    Loads the SCROLLS QASPER dataset.

    Args:
        split: The dataset split (default: "train").

    Returns:
        Hugging Face Dataset instance.
    """
    return load_dataset("tau/scrolls", "qasper", split=split)
