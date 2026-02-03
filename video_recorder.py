"""
Video Recorder for Wandb Logging (True Headless)

Records policy rollouts as videos and logs them to Weights & Biases.
Uses Newton's headless rendering - no display required.

Key insight from Newton conventions:
- Set PYGLET_HEADLESS=1 before imports
- Create ViewerGL(headless=True)
- Attach to existing training model (not a copy)
- Use training's live state to render frames

Usage:
    recorder = HeadlessVideoRecorder(env, agent, device)
    recorder.record_and_log(iteration, num_episodes=1, max_steps=300)
"""

import os
# CRITICAL: Must set before importing pyglet/newton
os.environ["PYGLET_HEADLESS"] = "1"

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import warp as wp
    import newton
    from newton.viewer import ViewerGL
    NEWTON_AVAILABLE = True
except ImportError:
    NEWTON_AVAILABLE = False


class HeadlessVideoRecorder:
    """
    True headless video recorder using Newton's ViewerGL(headless=True).

    This creates a separate headless viewer that attaches to the training
    environment's existing model and state - no display required.
    """

    def __init__(self, env, agent, device, width=640, height=480, fps=30):
        """
        Initialize the headless video recorder.

        Args:
            env: MimicKit environment
            agent: Trained agent with policy
            device: Computation device
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second for video
        """
        self._env = env
        self._agent = agent
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps

        # Headless viewer (created on first use)
        self._viewer = None
        self._initialized = False

    def _ensure_viewer(self):
        """Create headless viewer and attach to training model."""
        if self._initialized:
            return self._viewer is not None

        self._initialized = True

        if not NEWTON_AVAILABLE:
            print("[HeadlessVideoRecorder] Newton not available")
            return False

        # Get the engine and model from environment
        engine = getattr(self._env, '_engine', None)
        if engine is None:
            print("[HeadlessVideoRecorder] No engine in environment")
            return False

        sim_model = getattr(engine, '_sim_model', None)
        if sim_model is None:
            print("[HeadlessVideoRecorder] No sim_model in engine")
            return False

        try:
            # Create headless viewer
            self._viewer = ViewerGL(
                width=self._width,
                height=self._height,
                headless=True
            )

            # Attach to the SAME model used by training
            self._viewer.set_model(sim_model)

            print(f"[HeadlessVideoRecorder] Created headless viewer {self._width}x{self._height}")
            return True

        except Exception as e:
            print(f"[HeadlessVideoRecorder] Failed to create viewer: {e}")
            self._viewer = None
            return False

    def _get_state(self):
        """Get the current simulation state from training environment."""
        engine = getattr(self._env, '_engine', None)
        if engine is None:
            return None

        sim_state = getattr(engine, '_sim_state', None)
        if sim_state is None:
            return None

        return getattr(sim_state, 'raw_state', None)

    def _capture_frame(self):
        """Capture a single frame from current state."""
        if self._viewer is None:
            return None

        state = self._get_state()
        if state is None:
            return None

        try:
            # Render the current state (Newton convention)
            self._viewer.begin_frame(0.0)
            self._viewer.log_state(state)
            self._viewer.end_frame()

            # Get frame as numpy array
            frame_wp = self._viewer.get_frame()
            frame_np = frame_wp.numpy()

            return frame_np.copy()

        except Exception as e:
            # Silent fail on frame capture errors
            return None

    def record_episode(self, max_steps=300):
        """
        Record a single episode.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            List of frames as numpy arrays (H, W, 3) uint8
        """
        if not self._ensure_viewer():
            return []

        frames = []

        # Reset environment
        obs = self._env.reset()
        if isinstance(obs, dict):
            obs = obs.get('obs', obs)

        for step in range(max_steps):
            # Get action from policy
            with torch.no_grad():
                if hasattr(self._agent, '_model'):
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    action = self._agent._model.eval_actor(obs_tensor)
                    if isinstance(action, tuple):
                        action = action[0]
                    action = action.cpu().numpy()
                else:
                    # Random action fallback
                    action = np.zeros(self._env.get_action_size())

            # Step environment
            obs, reward, done, info = self._env.step(action)
            if isinstance(obs, dict):
                obs = obs.get('obs', obs)

            # Capture frame from current state
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)

            # Check if episode done (use first env if batched)
            if isinstance(done, np.ndarray):
                if done[0]:
                    break
            elif done:
                break

        return frames

    def frames_to_video(self, frames):
        """
        Convert frames to video array for wandb.

        Args:
            frames: List of numpy arrays (H, W, 3)

        Returns:
            numpy array (T, H, W, C) for wandb.Video
        """
        if not frames:
            return None

        # Stack frames: (T, H, W, C)
        video = np.stack(frames, axis=0)
        return video

    def record_and_log(self, iteration, num_episodes=1, max_steps=300, prefix="policy"):
        """
        Record episodes and log video to wandb.

        Args:
            iteration: Current training iteration (for labeling)
            num_episodes: Number of episodes to record
            max_steps: Maximum steps per episode
            prefix: Prefix for wandb log key
        """
        if not WANDB_AVAILABLE:
            print("[HeadlessVideoRecorder] wandb not available")
            return

        if wandb.run is None:
            print("[HeadlessVideoRecorder] No active wandb run")
            return

        all_frames = []

        for ep in range(num_episodes):
            frames = self.record_episode(max_steps=max_steps)
            all_frames.extend(frames)

        if not all_frames:
            print(f"[HeadlessVideoRecorder] No frames captured at iteration {iteration}")
            return

        # Convert to video
        video = self.frames_to_video(all_frames)

        if video is not None:
            # Log to wandb
            wandb.log({
                f"{prefix}_video": wandb.Video(
                    video,
                    fps=self._fps,
                    format="mp4"
                )
            }, step=iteration)

            print(f"[HeadlessVideoRecorder] Logged video at iteration {iteration} "
                  f"({len(all_frames)} frames, {video.shape})")


# Backwards compatibility alias
VideoRecorder = HeadlessVideoRecorder
SimpleVideoRecorder = HeadlessVideoRecorder
