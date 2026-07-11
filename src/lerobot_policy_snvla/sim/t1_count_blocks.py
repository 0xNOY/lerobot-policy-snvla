"""T1 counting task: put N identical blocks into the basket (spec P5, task T1).

BDDLはlibero_objectの実ファイル（floorシーン）をテンプレート化して生成する。
同一カテゴリの複数インスタンス（libero_spatialに前例あり）でPerceptual Aliasingを作る。
"""

import os
from pathlib import Path

import numpy as np

BASKET_BODY = "basket_1_main"
BLOCK_BODY_TEMPLATE = "{category}_{i}_main"
# 把持しやすく（高さ0.1で指がかかる）、footprint(半径0.025)が小さいので
# かご(内寸half≈0.064)に複数個置いても衝突しにくい
DEFAULT_CATEGORY = "chocolate_pudding"


def category_display_name(category: str) -> str:
    """BDDLカテゴリ名を実況・タスク指示用の表示名に変換する。"""
    return category.replace("_", " ")

# problem名はLIBEROの登録済みシーンクラス（TASK_MAPPING）に一致させる必要がある
_BDDL_TEMPLATE = """(define (problem LIBERO_Floor_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}
      (bin_region
          (:target floor)
          (:ranges (
              ({bin_range})
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


# ブロックのspawn域（ロボット手前側）とかごのspawn域（奥側）。
# いずれもエキスパートの到達実績がある範囲に収める。
BLOCK_SPAWN_X = (-0.20, 0.20)
BLOCK_SPAWN_Y = (-0.30, -0.12)
BLOCK_MIN_DIST = 0.10  # ブロック中心間の最小距離（footprint+把持クリアランス）
BASKET_SPAWN_X = (-0.08, 0.08)
BASKET_SPAWN_Y = (0.22, 0.28)


def sample_layout(n_blocks: int, rng: np.random.Generator) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """ブロックN個の中心座標とかごの中心座標をサンプルする（棄却法で最小距離を保証）。"""
    centers: list[tuple[float, float]] = []
    while len(centers) < n_blocks:
        x = rng.uniform(*BLOCK_SPAWN_X)
        y = rng.uniform(*BLOCK_SPAWN_Y)
        if all(np.hypot(x - cx, y - cy) >= BLOCK_MIN_DIST for cx, cy in centers):
            centers.append((x, y))
    basket = (rng.uniform(*BASKET_SPAWN_X), rng.uniform(*BASKET_SPAWN_Y))
    return centers, basket


def make_t1_bddl(
    n_blocks: int,
    out_dir: Path,
    object_category: str = DEFAULT_CATEGORY,
    language: str | None = None,
    seed: int | None = None,
) -> Path:
    """T1のBDDLを生成する。seedを与えるとエピソード固有のランダム配置になる。"""
    objs = object_names(n_blocks, object_category)
    if language is None:
        language = f"put {n_blocks} {category_display_name(object_category)}s into the basket"
    rng = np.random.default_rng(0 if seed is None else seed)
    centers, (bx, by) = sample_layout(n_blocks, rng)
    regions, init, goal = [], [], []
    for obj, (x, y) in zip(objs, centers, strict=True):
        region = f"{obj}_region"
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
        bin_range=f"{bx - 0.01} {by - 0.01} {bx + 0.01} {by + 0.01}",
        # パーサは「x1 x2 ... - category」の1行グループ形式のみ正しく扱う（行分割すると上書きされる）
        objects=f"    {' '.join(objs)} - {object_category}",
        obj_of_interest="\n".join(f"    {o}" for o in objs),
        init="\n".join(init),
        goal="\n".join(goal),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if seed is None else f"_s{seed}"
    path = out_dir / f"t1_count_blocks_{object_category}_n{n_blocks}{suffix}.bddl"
    path.write_text(text)
    return path


def make_t1_env(
    n_blocks: int,
    seed: int,
    camera_hw: int = 256,
    out_dir: Path | None = None,
    object_category: str = DEFAULT_CATEGORY,
):
    """T1環境を構築する。seedは物体配置（BDDL生成）とenv両方に適用される。"""
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    if out_dir is None:
        out_dir = Path.home() / ".cache" / "snvla_sim" / "bddl"
    bddl = make_t1_bddl(n_blocks, out_dir, object_category, seed=seed)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=camera_hw,
        camera_widths=camera_hw,
    )
    env.seed(seed)
    return env
