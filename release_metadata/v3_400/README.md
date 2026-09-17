# v3 400-scene training-ready release

This release contains the 400 accepted scenes currently completed from the
additive v3 batch generator. Each scene has one Clear target and three native
Cycles degraded inputs (`mild`, `medium`, `strong`). The preliminary batch
uses one lighting mode per scene; it is not a three-light same-scene control
set. A later supplement may add the three-light variants.

The training ZIP volumes contain 16-bit RGB PNG inputs/targets, one shared GT
bundle per scene, and `pairs.jsonl`. The uncompressed source output remains
local under the v3 workspace; per-scene `.blend` files were not saved.
