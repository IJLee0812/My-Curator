"""Shim — moved to my_curator.adapters.gst.utils.  Removed in R-7.

Mirrors every module-level attribute (public and private) so historical
callers — including tests that import ``_bbox_to_zone`` — keep working.
"""

from my_curator.adapters.gst import utils as _utils

for _name in dir(_utils):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_utils, _name)

del _utils, _name
