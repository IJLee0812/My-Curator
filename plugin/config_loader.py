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

"""Shim — moved to my_curator.adapters.gst.config_loader.  Removed in R-7.

The Singleton (``_config_instance``) lives in the new module; calls through
``get_config`` / ``reload_config`` here go through the function objects that
own that namespace, so priming order is preserved regardless of which path
the importer uses.
"""

from my_curator.adapters.gst.config_loader import (  # noqa: F401
    Config,
    get_config,
    reload_config,
)

__all__ = ["Config", "get_config", "reload_config"]
