from __future__ import annotations

from lerobot_snvla import register


def train() -> None:
    register()
    from lerobot.scripts.lerobot_train import main

    main()


def eval() -> None:
    register()
    from lerobot.scripts.lerobot_eval import main

    main()


def dataset_viz() -> None:
    register()
    from lerobot.scripts.lerobot_dataset_viz import main

    main()
