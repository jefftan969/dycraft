<!-- Mirrored from https://nvlabs.github.io/curobo/latest/getting-started/inverse_kinematics.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# Inverse Kinematics

Solve inverse kinematics on the GPU with multi-seed optimization.

Inverse kinematics (IK) finds the joint angles that place a robot’s end-effector at a desired 6-DOF pose. cuRobo formulates IK as a nonlinear optimization problem and solves many random seeds in parallel on the GPU, achieving high success rates and sub-millimeter accuracy in a single call. The same solver supports batched targets, scene-aware collision avoidance, self-collision checking, and runtime world updates.

By the end of this tutorial you will have:

- Solved IK for a single target pose and verified the result with FK

- Solved IK for 100 target poses in a single batched call

- Added obstacles and self-collision checking for collision-free IK

- Updated the obstacle scene at runtime without recreating the solver

- Launched an interactive Viser viewer for real-time drag-and-solve IK

- Visualized a 3D reachability map as a colored point cloud

## Step 1: Run the tutorial

    python -m curobo.examples.getting_started.inverse_kinematics

This runs single, batched, and collision-free IK in sequence. Three interactive visualization modes are also available:

    python -m curobo.examples.getting_started.inverse_kinematics --visualize

Full IK (LM + LBFGS): drag the end-effector gizmo to solve IK in real time; drag obstacles to see collision avoidance adapt.

    python -m curobo.examples.getting_started.inverse_kinematics --differential

Differential IK: smooth reactive control that tracks the gizmo continuously, suitable for teleoperation-style interaction.

    python -m curobo.examples.getting_started.inverse_kinematics --reachability

To try reachability with a dual-arm robot:

    python -m curobo.examples.getting_started.inverse_kinematics --reachability --robot dual_ur10e.yml

Reachability map: a 100x100 grid of IK queries is solved on a 2D slice plane and displayed as an image (green = reachable, red = unreachable). Drag the gizmo to move or rotate the plane; use the slider to resize it.

All three modes open at <a href="http://localhost:8080" class="reference external">http://localhost:8080</a> in your browser.

## Step 2: Check the output

When the tutorial finishes successfully you will see:

    === Single IK ===
    IK solved!
      Joint angles: tensor(...)
      Position error: 0.XXX mm

    === Batched IK (100 poses) ===
    Solved 95/100 poses (95% success)
    Mean position error: 0.XXX mm
    Max position error:  0.XXX mm

    === Collision-Free IK ===
    Single collision-free IK solved!
      Position error: 0.XXX mm

    After adding obstacle -- still solved!
      Position error: 0.XXX mm

    Batched collision-free IK: 48/50 solved

## Step 3: Understand the pipeline

The tutorial demonstrates four levels of IK solving:

1.  **Single IK**: Create an `InverseKinematics` with `num_seeds=32`, define a target `Pose` (position + quaternion in wxyz order), and call `solve_pose`. The solver runs 32 parallel optimization seeds and returns the best solution. Verify by running FK on the solution and checking the position error.

2.  **Batched IK**: Pass a `Pose` with shape `(B,`` ``3)` positions and `(B,`` ``4)` quaternions to `solve_pose`. All B targets are solved in parallel on the GPU, useful for reachability analysis or grasp pose evaluation.

3.  **Collision-free IK**: Pass `scene_model` and `self_collision_check=True` when creating the solver config. The optimizer adds collision cost terms that keep every robot link sphere clear of obstacles and other links.

4.  **Runtime world updates**: Call `update_world` with a new `Scene` to add, remove, or move obstacles without recreating the solver.

## Step 4: Interactive IK with Viser (advanced)

All three interactive modes launch a <a href="https://viser.studio" class="reference external">Viser</a> web viewer that renders the robot, its collision spheres, and scene obstacles.

- **Full IK** (`--visualize`): Uses the complete LM + LBFGS optimization pipeline. A 6-DOF gizmo lets you drag the end-effector target; the solver re-runs on every gizmo update. Obstacles are interactive – drag them to new positions and the solver automatically avoids them.

- **Differential IK** (`--differential`): Smooth, continuous tracking where each solve warm-starts from the previous solution, producing small joint-space steps ideal for reactive control and teleoperation.

- **Reachability map** (`--reachability`): Solves a dense grid of IK queries on a 2D slice plane and visualizes success as a colored point cloud (green = reachable, red = unreachable). Drag the gizmo to reposition or rotate the plane; use the slider to resize it.

## Step 5: Reachability analysis with batched IK

A common use of batched IK is **reachability analysis**: given a robot configuration and scene, which end-effector poses can the robot actually reach? cuRobo’s `solve_pose` solves thousands of IK queries in a single GPU call, making it practical to sweep a dense grid of candidate poses and classify each as reachable or unreachable.

The `--reachability` mode demonstrates this by sampling a 100x100 grid of positions on a 2D slice plane, solving IK for every grid point, and coloring the result (green = solved, red = failed). Because the solver is collision-aware, the reachability map automatically reflects obstacles and self-collision constraints – adding an obstacle carves out an unreachable region in real time.

This workflow is useful for:

- **Workcell design**: verify that a robot can reach all required task poses before committing to a layout.

- **Grasp filtering**: pre-screen candidate grasp poses and discard unreachable ones before running the full motion planner.

- **Multi-arm coverage**: with robots like `dual_ur10e.yml`, visualize the overlapping workspace of two arms to plan handoff zones.
