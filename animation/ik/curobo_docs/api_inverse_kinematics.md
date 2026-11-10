<!-- Mirrored from https://nvlabs.github.io/curobo/latest/api/curobo.inverse_kinematics.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# curobo.inverse_kinematics module

Inverse kinematics solver module.

This module provides collision-aware inverse kinematics solving with optimization.

Example

[\`\`](#id1)[\`](#id3)python from curobo import InverseKinematics, InverseKinematicsCfg

config = InverseKinematicsCfg.create(
robot=”franka.yml”, num_seeds=12,

) ik = InverseKinematics(config) result = ik.solve_pose(goal_tool_poses=target_poses) [\`\`](#id5)[\`](#id7)

InverseKinematics
alias of `curobo._src.solver.solver_ik.IKSolver`

InverseKinematicsCfg
alias of `curobo._src.solver.solver_ik_cfg.IKSolverCfg`

InverseKinematicsResult
alias of `curobo._src.solver.solver_ik_result.IKSolverResult`
