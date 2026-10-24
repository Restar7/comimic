"""MOSAIC mechanisms ported onto Isaac-GR00T N1.7 (SONIC78 route).

The package is additive: it never changes upstream ``gr00t`` behaviour when its
entry points are unused.  It owns four things:

* ``contracts`` -- the versioned SONIC78 action/state/time/provenance contracts;
* ``policy``   -- K-candidate sampling on the native N1.7 flow sampler, decoding,
  the decision gate and the shadow runner;
* ``ac``/``stage`` -- the typed consequence model and the stage frontier used to
  score candidates;
* ``posttrain``/``data``/``eval`` -- verified-teacher materialisation, the
  post-training launcher, and the offline evaluation tools.

See ``../mosaic_n17/README.md`` and
``docs/projects/mosaic-n17-migration-plan-20261022.md`` in the FlowBody
workspace for the migration specification this package implements.
"""

from __future__ import annotations


__all__ = ["__version__"]

__version__ = "0.1.0"
