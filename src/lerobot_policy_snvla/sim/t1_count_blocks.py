"""T1 counting task: put N identical blocks into the basket (spec P5, task T1).

BDDLはlibero_objectの実ファイル（floorシーン）をテンプレート化して生成する。
同一カテゴリの複数インスタンス（libero_spatialに前例あり）でPerceptual Aliasingを作る。
"""

import os
from pathlib import Path

T1_TASK_DESCRIPTION_TEMPLATE = "put {n} blocks into the basket"
BASKET_BODY = "basket_1_main"
BLOCK_BODY_TEMPLATE = "{category}_{i}_main"
# 把持しやすく（高さ0.1で指がかかる）、footprint(半径0.025)が小さいので
# かご(内寸half≈0.064)に複数個置いても衝突しにくい
DEFAULT_CATEGORY = "chocolate_pudding"

# problem名はLIBEROの登録済みシーンクラス（TASK_MAPPING）に一致させる必要がある
_BDDL_TEMPLATE = """(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}
      (bin_region
          (:target floor)
          (:ranges (
              (-0.01 0.25 0.01 0.27)
            )
          )
      )
      (contain_region
          (:target basket_1)
      )
    )

  (:fixtures
    floor - floor
  )

  (:objects
{objects}
    basket_1 - basket
  )

  (:obj_of_interest
{obj_of_interest}
    basket_1
  )

  (:init
{init}
    (On basket_1 floor_bin_region)
  )

  (:goal
    (And
{goal}
    )
  )

)
"""


def object_names(n_blocks: int, category: str = DEFAULT_CATEGORY) -> list[str]:
    return [f"{category}_{i + 1}" for i in range(n_blocks)]


def object_body_names(n_blocks: int, category: str = DEFAULT_CATEGORY) -> list[str]:
    return [BLOCK_BODY_TEMPLATE.format(category=category, i=i + 1) for i in range(n_blocks)]


def _block_spawn_xy(i: int) -> tuple[float, float]:
    """Spread blocks on a grid in front of the robot (y < 0 half of the floor)."""
    x = -0.15 + 0.15 * (i % 3)
    y = -0.25 + 0.12 * (i // 3)
    return x, y


def make_t1_bddl(n_blocks: int, out_dir: Path, object_category: str = DEFAULT_CATEGORY) -> Path:
    objs = object_names(n_blocks, object_category)
    language = T1_TASK_DESCRIPTION_TEMPLATE.format(n=n_blocks)
    regions, init, goal = [], [], []
    for i, obj in enumerate(objs):
        region = f"{obj}_region"
        x, y = _block_spawn_xy(i)
        regions.append(
            f"      ({region}\n"
            f"          (:target floor)\n"
            f"          (:ranges (\n"
            f"              ({x - 0.025} {y - 0.025} {x + 0.025} {y + 0.025})\n"
            f"            )\n"
            f"          )\n"
            f"      )"
        )
        init.append(f"    (On {obj} floor_{region})")
        goal.append(f"      (In {obj} basket_1_contain_region)")
    text = _BDDL_TEMPLATE.format(
        language=language,
        regions="\n".join(regions),
        # パーサは「x1 x2 ... - category」の1行グループ形式のみ正しく扱う（行分割すると上書きされる）
        objects=f"    {' '.join(objs)} - {object_category}",
        obj_of_interest="\n".join(f"    {o}" for o in objs),
        init="\n".join(init),
        goal="\n".join(goal),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"t1_count_blocks_n{n_blocks}.bddl"
    path.write_text(text)
    return path


def make_t1_env(n_blocks: int, seed: int, camera_hw: int = 256, out_dir: Path | None = None):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    if out_dir is None:
        out_dir = Path.home() / ".cache" / "snvla_sim" / "bddl"
    bddl = make_t1_bddl(n_blocks, out_dir)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=camera_hw,
        camera_widths=camera_hw,
    )
    env.seed(seed)
    return env
