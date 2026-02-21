"""Record a video of a trained MimicKit policy and save as MP4."""

import sys
import os
import argparse
import numpy as np
import torch

CODESIGN_DIR = os.path.dirname(os.path.abspath(__file__))
MIMICKIT_DIR = os.path.join(CODESIGN_DIR, "..", "MimicKit")
MIMICKIT_SRC = os.path.join(MIMICKIT_DIR, "mimickit")


def main():
    parser = argparse.ArgumentParser(description="Record policy video")
    parser.add_argument("--model_file", required=True, help="Path to .pt model file")
    parser.add_argument("--output", default="policy_video.mp4", help="Output MP4 path")
    parser.add_argument("--num_steps", type=int, default=300, help="Steps to record")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_envs", type=int, default=1)
    args = parser.parse_args()

    import warp as wp
    import newton  # noqa: F401
    from newton.viewer import ViewerGL

    if MIMICKIT_SRC not in sys.path:
        sys.path.insert(0, MIMICKIT_SRC)

    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import learning.base_agent as base_agent_mod
    import util.mp_util as mp_util
    import util.util as util

    # Config paths
    data_dir = os.path.join(MIMICKIT_DIR, "data")
    env_config = os.path.join(data_dir, "envs", "amp_g1_env.yaml")
    engine_config = os.path.join(data_dir, "engines", "newton_engine.yaml")
    agent_config = os.path.join(data_dir, "agents", "amp_g1_agent.yaml")

    device = "cuda:0"

    # Init distributed (single process)
    mp_util.init(0, 1, device, 6000)
    util.set_rand_seed(np.uint64(42))

    # Build env and agent (visualize=False — we use our own viewer)
    os.chdir(MIMICKIT_DIR)
    env = env_builder.build_env(env_config, engine_config, args.num_envs, device, False)
    agent = agent_builder.build_agent(agent_config, env, device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent_mod.AgentMode.TEST)
    print(f"Loaded model from {args.model_file}")

    # MimicKit's import chain sets PYGLET_HEADLESS=1 (video_recorder.py) and then
    # newton_engine.py does `import pyglet` which caches headless=True in pyglet.options.
    # We must override BOTH the env var and pyglet's cached option before ViewerGL
    # creates a window, since pyglet.gl (which reads the option) hasn't been imported yet.
    os.environ.pop("PYGLET_HEADLESS", None)
    import pyglet
    pyglet.options["headless"] = False

    # Create windowed viewer for frame capture
    engine = env._engine
    sim_model = engine._sim_model
    viewer = ViewerGL(headless=False, width=args.width, height=args.height)
    viewer.set_model(sim_model, max_worlds=1)
    viewer.set_camera(pos=wp.vec3(3.0, -3.0, 1.5), pitch=-10.0, yaw=90.0)
    print(f"Viewer created: {args.width}x{args.height}")

    # Run episode and capture frames
    obs, info = env.reset()
    frames = []

    with torch.no_grad():
        for step in range(args.num_steps):
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            wp.synchronize()

            # Capture frame
            state = getattr(engine._sim_state, 'raw_state',
                            getattr(engine._sim_state, 'state', None))
            if state is not None:
                viewer.begin_frame(0.0)
                viewer.log_state(state)
                viewer.end_frame()
                frame = viewer.get_frame()
                if frame is not None:
                    frames.append(frame.numpy().copy())

            if step % 50 == 0:
                print(f"  Step {step}/{args.num_steps} ({len(frames)} frames)")

    print(f"Captured {len(frames)} frames")

    if not frames:
        print("No frames captured!")
        return

    # Save as MP4
    try:
        import imageio
    except ImportError:
        print("Install imageio: pip install imageio imageio-ffmpeg")
        return

    video = np.stack(frames, axis=0)
    if video.dtype != np.uint8:
        video = np.clip(video * 255, 0, 255).astype(np.uint8)

    imageio.mimwrite(args.output, video, fps=args.fps, codec='libx264')
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
