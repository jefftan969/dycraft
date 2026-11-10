<!-- Mirrored from https://nvlabs.github.io/curobo/latest/getting-started/build_robot_model.html
     cuRobo v0.8.0 (cuRoboV2) docs, NVIDIA, Apache-2.0.
     Local copy because a dycraft run has no network.
     Refresh: python animation/ik/fetch_curobo_docs.py animation/ik/curobo_docs -->

# Build Robot Model

Build a cuRobo robot configuration from a URDF file.

cuRobo needs two pieces of information that a standard URDF does not contain: **collision spheres** fitted to each link mesh and a **self-collision ignore matrix** that lists link pairs whose collisions can be safely skipped. The <a href="../api/curobo.robot_builder.html#curobo.robot_builder.RobotBuilder" class="reference internal" title="curobo.robot_builder.RobotBuilder"><code class="sourceCode python">RobotBuilder</code></a> API generates both and writes them to a YAML or XRDF config file consumed by every downstream cuRobo module (forward kinematics, inverse kinematics, motion planning, MPC).

<a href="../_images/franka_spheres.png" class="reference internal image-reference"><img src="../_images/franka_spheres.png" style="width: 100%;" alt="Robot mesh (left) and its collision sphere approximation (right)" /></a>

By the end of this tutorial you will have:

- Created a cuRobo robot configuration from a URDF file

- Fitted collision spheres to each link mesh using the MorphIt optimizer

- Computed an optimized self-collision ignore matrix

- Inspected per-link sphere fit quality metrics

- Visualized the fitted spheres in a browser

- Saved the configuration in both YAML and XRDF formats

## Step 1: Prepare your URDF

You need a URDF file and a directory containing the referenced mesh assets. cuRobo ships robot URDFs under `curobo/content/assets/robot/`, so the tutorial works out of the box with the bundled Franka Panda. For your own robot, `package://` prefixes in mesh filenames are stripped automatically; set `--asset-path` to the directory that contains the remaining relative path (e.g., if the URDF says `package://my_robot/meshes/link.stl`, point `--asset-path` at the parent of `my_robot/`).

If your robot needs a **floating base** (e.g., a humanoid whose pelvis moves freely in space), use `extra_links` with `child_link_name` in the YAML config to insert virtual joints between `base_link` and the robot’s root body. This requires no URDF modification. See the <a href="humanoid_retargeting.html#humanoid-retargeting" class="reference internal">humanoid retargeting guide</a> for a complete example using the Unitree G1.

## Step 2: Run the tutorial

    python -m curobo.examples.getting_started.build_robot_model \
        --urdf curobo/content/assets/robot/franka_description/franka_panda.urdf \
        --asset-path curobo/content/assets/robot/franka_description \
        --output franka_custom.yml \
        --clip-link panda_link0 z 0.0 \
        --compute-metrics

Add `--visualize` to inspect the fitted spheres in a <a href="https://viser.studio" class="reference external">Viser</a> viewer at `http://localhost:8080`. `--clip-link`` ``panda_link0`` ``z`` ``0.0` keeps the base link’s spheres from protruding below the mounting surface (see <a href="#step-5-tuning-advanced" class="reference internal">Step 5</a>).

## Step 3: Check the output

When the tutorial finishes successfully you will see:

    Building robot model from URDF: ...franka_panda.urdf
    Found 11 links in robot

    Fitting collision spheres...
    Fitted 87 spheres across 9 links

      link                      n_sph  cover%  protr%   prot_mm  gap_mm  vol_ratio
      ---------------------------------------------------------------------------
      panda_link0                  12   98.3%    4.1%     0.72mm   1.04mm     1.124
      ...

    Computing collision matrix...
    Created collision ignore matrix with 28 entries

    Saving to: franka_custom.yml
    ✓ Robot model created successfully!

The generated `franka_custom.yml` (or `.xrdf` if you chose that extension) is ready to use with `ik_solver`, `motion_gen`, or any other cuRobo wrapper.

## Step 4: Understand the pipeline

The builder runs three stages:

1.  **Sphere fitting**: Each link mesh is approximated by a set of spheres using the MorphIt optimizer (an Adam-based iterative fit that balances interior *coverage* against surface *protrusion*). The `--sphere-density` multiplier controls how many spheres are allocated per link.

2.  **Self-collision matrix**: Link pairs that are always in collision (e.g. adjacent joints) or never reachable are identified via random joint sampling (`--num-collision-samples`) and placed in an ignore set so the downstream planner skips them.

3.  **Export**: The sphere and matrix data, together with the kinematic tree, are serialized to YAML (native cuRobo) or XRDF (Isaac Sim / Isaac Lab). Use `--export-xrdf` to emit both formats at once.

## Step 5: Tuning (advanced)

- `--coverage-weight` / `--protrusion-weight` control the MorphIt loss balance. Raise `coverage-weight` (default 1000) for tighter volume filling; raise `protrusion-weight` (default 10) to reduce overshoot.

- `--sphere-density`` ``2.0` doubles the sphere budget per link.

- `--edit-config`` ``existing.yml`` ``--refit-link`` ``panda_hand` re-fits spheres for a single link without re-running the entire pipeline.

- `--seed`` ``42` pins NumPy and PyTorch RNGs for reproducible results.

- `--clip-link`` ``base_link`` ``z`` ``0.0` prevents spheres on `base_link` from extending below `z=0` in link-local coordinates. This is useful when the robot is mounted on a stand or bolted to the floor – without clipping, the base link spheres may overlap the mounting surface and cause perpetual collisions. The constraint is enforced as both a differentiable MorphIt loss and a hard post-fit clamp. Can be repeated for multiple links.

Once you have run the tutorial, open <a href="#module-curobo.examples.getting_started.build_robot_model" class="reference internal" title="curobo.examples.getting_started.build_robot_model"><code class="sourceCode python">curobo.examples.getting_started.build_robot_model</code></a> in your editor. The inline comments walk through the key design decisions: how sphere density affects collision checking fidelity, when to raise or lower the MorphIt loss weights, and how the self-collision matrix is pruned.
