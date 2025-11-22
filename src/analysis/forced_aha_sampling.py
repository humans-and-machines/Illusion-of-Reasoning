#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilities to prepare forced-aha samples for the forced-aha analysis."""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import pandas as pd

def prepare_forced_aha_samples(
    args,
    load_root_samples: Callable[
        [str, Optional[str], str, str, Optional[str]], pd.DataFrame
    ],
    load_single_root_samples: Callable[
        [str, Optional[str], str, Optional[str]], Tuple[pd.DataFrame, pd.DataFrame]
    ],
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Create output directory and load pass1/pass2 samples for the forced-aha analysis.
    """
    suffix = ""
    if args.pass2_key and args.pass2_key != "pass2":
        suffix = f"_{args.pass2_key}"
    out_dir = args.out_dir or os.path.join(args.root1, f"forced_aha_effect{suffix}")
    os.makedirs(out_dir, exist_ok=True)

    if args.root2:
        df1 = load_root_samples(
            args.root1,
            args.split,
            variant="pass1",
            entropy_field=args.entropy_field,
            pass2_key=None,
        )
        df2 = load_root_samples(
            args.root2,
            args.split,
            variant="pass2",
            entropy_field=args.entropy_field,
            pass2_key=args.pass2_key,
        )
    else:
        df1, df2 = load_single_root_samples(
            args.root1,
            args.split,
            entropy_field=args.entropy_field,
            pass2_key=args.pass2_key,
        )
    return out_dir, df1, df2
