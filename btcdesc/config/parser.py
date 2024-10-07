# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import sys
import importlib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class BTCDescConfig(BaseModel):
    # for submap process
    cloud_ds_size: float = 0.25

    # for binary descriptor
    useful_corner_num: int = 30
    plane_merge_normal_thre: float
    plane_merge_dis_thre: float
    plane_detection_thre: float = 0.01
    voxel_size: float = 1.0
    voxel_init_num: int = 10
    proj_plane_num: int = 1
    proj_image_resolution: float = 0.5
    proj_image_high_inc: float = 0.5
    proj_dis_min: float = 0
    proj_dis_max: float = 5
    summary_min_thre: float = 10
    line_filter_enable: int = 0

    # for triangle descriptor
    descriptor_near_num: float = 10
    descriptor_min_len: float = 1
    descriptor_max_len: float = 10
    non_max_suppression_radius: float = 3.0
    std_side_resolution: float = 0.2

    # for place recognition
    skip_near_num: int = 20
    candidate_num: int = 50
    sub_frame_num: int = 10
    rough_dis_threshold: float = 0.03
    similarity_threshold: float = 0.7
    icp_threshold: float = 0.5
    normal_threshold: float = 0.1
    dis_threshold: float = 0.3


def load_config(config_file: Optional[Path]) -> BTCDescConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = None
    if config_file is not None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            print(
                "[ERROR] Custom configuration file specified but PyYAML is not installed on your system,"
                " run `pip install pyyaml`"
            )
            sys.exit(1)
        with open(config_file) as cfg_file:
            config = yaml.safe_load(cfg_file)
        return BTCDescConfig(**config)
    else:
        return BTCDescConfig()


def write_config(config: BTCDescConfig, filename: str):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
