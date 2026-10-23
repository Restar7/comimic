#!/usr/bin/env python
"""Start the real GR00T PolicyServer (same entrypoint the deployment uses).

    python run_server.py --model-path ckpt_sirius_50k \
        --embodiment-tag UNITREE_G1_SONIC --device cpu --host 127.0.0.1 --port 5550

The only addition over `python gr00t/eval/run_gr00t_server.py` is
`meta_patch.install()`, which keeps the throw-away random-init copy of the VLM
backbone on the meta device (the real weights come from the checkpoint shards).
"""

import meta_patch

meta_patch.install()

import tyro  # noqa: E402
from gr00t.eval.run_gr00t_server import ServerConfig, main  # noqa: E402

if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
