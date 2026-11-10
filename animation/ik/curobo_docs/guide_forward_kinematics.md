<!-- Mirrored from https://nvlabs.github.io/curobo/latest/getting-started/forward_kinematics.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# Forward Kinematics

Compute forward kinematics on the GPU with automatic differentiation.

Forward kinematics (FK) maps a vector of joint angles to the 6-DOF pose of every link in the kinematic chain. cuRobo evaluates FK entirely on the GPU using a parallel product-of-exponentials formulation, and every operation is differentiable through PyTorch autograd. This means FK can serve as a building block inside larger optimization loops (IK, motion planning, MPC) with gradients flowing back to the joint angles at zero extra cost.

By the end of this tutorial you will have:

- Loaded a robot model from a cuRobo YAML configuration

- Computed the end-effector pose for a single joint configuration

- Evaluated FK for 1 000 configurations in a single batched call

- Back-propagated a position loss through FK to obtain joint-angle gradients

## Step 1: Run the tutorial

    python -m curobo.examples.getting_started.forward_kinematics

## Step 2: Check the output

When the tutorial finishes successfully you will see:

    Robot has 7 degrees of freedom
    Tool frames: ['panda_hand']

    Single FK:
      EE position: tensor(...)
      EE quaternion (wxyz): tensor(...)

    Batched FK (1000 configs): 0.XX ms
      EE positions shape: torch.Size([1000, 1, 3])

    Differentiable FK:
      Gradient w.r.t. joints: tensor(...)

## Step 3: Understand the pipeline

The example demonstrates three capabilities:

1.  **Single FK**: Pass one joint configuration and receive the end-effector position and orientation (quaternion in wxyz order). The result also contains poses for every tool frame defined in the robot config.

2.  **Batched FK**: Pass a `(B,`` ``num_dof)` tensor of joint configurations and receive `(B,`` ``num_frames,`` ``3)` positions and `(B,`` ``num_frames,`` ``4)` quaternions. All configurations are evaluated in parallel on the GPU, making large-scale reachability analysis fast.

3.  **Differentiable FK**: Because FK is implemented as standard PyTorch operations, calling `.backward()` on any scalar derived from the output produces gradients with respect to the input joint angles. This is the foundation of cuRobo’s optimization-based IK and motion planning.
