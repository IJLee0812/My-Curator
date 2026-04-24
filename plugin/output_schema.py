###################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###################################################################################################

"""Pydantic schema for VLM output validation.

Mirrors the JSON shape documented in configs/config_driving_scene.yaml
system_prompt. Used to flag malformed outputs via metadata.json_valid in
the Kafka message; invalid outputs are still published.
"""

from pydantic import BaseModel, ConfigDict


class RoadFeatures(BaseModel):
    model_config = ConfigDict(extra="allow")

    num_lanes: int
    lane_markings: str
    road_surface: str
    road_condition: str


class KeyObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    description: str


class EgoVehicle(BaseModel):
    model_config = ConfigDict(extra="allow")

    action: str
    estimated_speed: str


class DrivingSceneResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    scene_summary: str
    road_type: str
    road_features: RoadFeatures
    weather: str
    visibility: str
    traffic_density: str
    key_objects: list[KeyObject]
    ego_vehicle: EgoVehicle
    potential_risks: list[str]
