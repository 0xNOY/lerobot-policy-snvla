import os

# Offscreen rendering backend for all sim tests. Pure-logic tests are unaffected.
os.environ.setdefault("MUJOCO_GL", "egl")
