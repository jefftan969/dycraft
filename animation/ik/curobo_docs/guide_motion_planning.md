<!-- Mirrored from https://nvlabs.github.io/curobo/latest/getting-started/motion_planning.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# Motion Planning

Plan collision-free trajectories and grasp motions with GPU-accelerated optimization.

Motion planning finds a smooth, collision-free path from a start joint configuration to a goal end-effector pose. cuRobo formulates this as a trajectory optimization problem: it parameterizes the path as a sequence of knot points and jointly optimizes them for smoothness (velocity, acceleration, jerk costs) and feasibility (collision avoidance, joint limits). Multiple trajectory seeds are optimized in parallel on the GPU, and the best collision-free result is returned.

cuRobo also supports **grasp planning**, which chains three trajectory segments – approach, grasp, and lift – into a single call. The planner first solves for a collision-free path to a pre-grasp pose (offset along the approach axis), then plans the final approach to the grasp pose with finger-link collisions disabled, and finally plans a lift motion away from the surface.

By the end of this tutorial you will have:

- Initialized a GPU-accelerated motion planner for the Franka Panda robot

- Defined a box obstacle in the planning scene

- Planned a collision-free trajectory from a start configuration to a goal pose

- Planned a three-phase grasp motion (approach, grasp, lift)

- Saved trajectory plots as PDF files

## Step 1: Run the tutorial

    python -m curobo.examples.getting_started.motion_planning

This runs both the pose-to-pose planner and the grasp planner in sequence. To run only one mode:

    python -m curobo.examples.getting_started.motion_planning --mode pose
    python -m curobo.examples.getting_started.motion_planning --mode grasp

## Step 2: Check the output

When the tutorial finishes successfully you will see:

    === Pose-to-Pose Motion Planning ===
    ✓ Planning succeeded!
    Trajectory has 250 waypoints
    Duration: 5.00s
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/motion_plan.pdf

    === Grasp Planning ===
    ✓ Grasp planning succeeded!
      Approach: 120 waypoints
      Grasp:    45 waypoints
      Lift:     60 waypoints
    Trajectory plot saved to: ~/.cache/curobo/examples/motion_planning/grasp_plan.pdf

Output files are written to `~/.cache/curobo/` by default. Override with `--output-dir` or by setting `curobo._src.runtime.cache_dir` in Python.

## Step 3: Understand the pipeline

**Pose-to-pose planning** walks through five stages:

1.  **Initialize MotionPlanner**: Load the robot config (`franka.yml`) and a scene (`collision_test.yml`). The `warmup()` call pre-compiles CUDA graphs for faster subsequent planning calls.

2.  **Set start and goal**: The start is a `JointState` (joint angles in radians), and the goal is a `Pose` (position + quaternion for the end-effector). Obstacle poses use the format `[x,`` ``y,`` ``z,`` ``qw,`` ``qx,`` ``qy,`` ``qz]` (meters, wxyz quaternion).

3.  **Plan**: `plan_pose` runs IK to find a goal configuration, then optimizes a trajectory across multiple seeds. It returns a result with `success`, the optimized trajectory, and an interpolated version.

4.  **Save**: The interpolated joint trajectory is plotted with `matplotlib` and saved as a PDF.

**Grasp planning** extends this with `plan_grasp`, which chains three trajectory segments:

1.  **Approach**: Plan from the current configuration to a pre-grasp pose, offset along `grasp_approach_axis` by `grasp_approach_offset` (default -15 cm along the tool Z axis). Finger-link collisions are disabled so the gripper can reach into tight spaces.

2.  **Grasp**: Plan the final approach from the pre-grasp pose to the grasp pose itself.

3.  **Lift**: Plan a retraction from the grasp pose, offset along `grasp_lift_axis` by `grasp_lift_offset`.

The planner accepts a *goal set* of candidate grasp poses via `ToolPose`, selects the most reachable one, and returns all three trajectory segments in a `GraspPlanResult`.

## Step 4: Interactive motion planning with Viser

For an interactive version with a web-based 3D viewer, run:

    python -m curobo.examples.getting_started.motion_planning --visualize

Open <a href="http://localhost:8080" class="reference external">http://localhost:8080</a> in your browser. Drag the target frame to set the goal pose, move obstacles, and click “Move” to plan and execute. Click “Grasp” to plan and execute a three-phase grasp motion.
