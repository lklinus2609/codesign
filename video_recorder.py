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

    def __init__(self, env, agent, device, width=640, height=480, fps=30, save_dir=None):
        """
        Initialize the headless video recorder.

        Args:
            env: MimicKit environment
            agent: Trained agent with policy
            device: Computation device
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second for video
            save_dir: Directory to save local video files (None = don't save locally)
        """
        self._env = env
        self._agent = agent
        self._device = device
        self._width = width
        self._height = height
        self._fps = fps
        self._save_dir = save_dir

        # Create save directory if specified
        if self._save_dir is not None:
            os.makedirs(self._save_dir, exist_ok=True)

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
            # Try to find engine in env.env if it's wrapped
            inner_env = getattr(self._env, 'env', None)
            if inner_env:
                engine = getattr(inner_env, '_engine', None)
        
        if engine is None:
            print(f"[HeadlessVideoRecorder] No _engine in environment (type: {type(self._env)})")
            # List available attributes to help debugging
            print(f"[HeadlessVideoRecorder] Available env attributes: {[a for a in dir(self._env) if not a.startswith('__')]}")
            return False

        sim_model = getattr(engine, '_sim_model', None)
        if sim_model is None:
            print(f"[HeadlessVideoRecorder] No _sim_model in engine (type: {type(engine)})")
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
            inner_env = getattr(self._env, 'env', None)
            if inner_env:
                engine = getattr(inner_env, '_engine', None)
        
        if engine is None:
            return None

        sim_state = getattr(engine, '_sim_state', None)
        if sim_state is None:
            return None

        # raw_state is a Newton convention
        state = getattr(sim_state, 'raw_state', None)
        if state is None:
            # Fallback for some environment versions
            state = getattr(sim_state, 'state', None)
            
        return state

    def _capture_frame(self):
        """Capture a single frame from current state."""
        if self._viewer is None:
            return None

        state = self._get_state()
        if state is None:
            # Only print once to avoid spamming
            if not hasattr(self, '_state_error_printed'):
                print("[HeadlessVideoRecorder] State is None in _capture_frame")
                self._state_error_printed = True
            return None

        try:
            # Render the current state (Newton convention)
            self._viewer.begin_frame(0.0)
            self._viewer.log_state(state)
            self._viewer.end_frame()

            # Get frame as numpy array
            frame_wp = self._viewer.get_frame()
            if frame_wp is None:
                return None
                
            frame_np = frame_wp.numpy()
            return frame_np.copy()

        except Exception as e:
            # Only print once to avoid spamming
            if not hasattr(self, '_capture_error_printed'):
                print(f"[HeadlessVideoRecorder] Frame capture exception: {e}")
                self._capture_error_printed = True
            return None

    def record_episode(self, max_steps=300):
        """
        Record a single episode.

        Uses the first environment from the batch for recording.
        The policy runs on all envs but we only track env 0.

        Args:
            max_steps: Maximum steps per episode

        Returns:
            List of frames as numpy arrays (H, W, 3) uint8
        """
        if not self._ensure_viewer():
            return []

        frames = []

        # Reset environment (returns batched obs for all envs)
        obs, info = self._env.reset()
        if isinstance(obs, dict):
            obs = obs.get('obs', obs)

        # Convert to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self._device)

        for step in range(max_steps):
            # Get action from policy (runs on all envs)
            with torch.no_grad():
                if hasattr(self._agent, '_model'):
                    # Ensure obs is properly batched tensor
                    if not isinstance(obs, torch.Tensor):
                        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
                    else:
                        obs_tensor = obs.float()

                    # Handle single obs case
                    if obs_tensor.dim() == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)

                    # Normalize observation like the agent does
                    if hasattr(self._agent, '_obs_norm'):
                        obs_tensor = self._agent._obs_norm.normalize(obs_tensor)

                    action_dist = self._agent._model.eval_actor(obs_tensor)
                    if isinstance(action_dist, tuple):
                        action_dist = action_dist[0]

                    # Extract action from distribution (mode = mean for Gaussian)
                    if hasattr(action_dist, 'mode'):
                        action = action_dist.mode
                    elif hasattr(action_dist, 'mean'):
                        action = action_dist.mean
                    else:
                        # Already a tensor
                        action = action_dist

                    # Unnormalize action if needed
                    if hasattr(self._agent, '_a_norm'):
                        action = self._agent._a_norm.unnormalize(action)
                else:
                    # Random action fallback
                    num_envs = obs.shape[0] if hasattr(obs, 'shape') and len(obs.shape) > 1 else 1
                    action = torch.zeros((num_envs, self._env.get_action_size()), device=self._device)

            # Step environment (all envs step together)
            obs, reward, done, info = self._env.step(action)
            if isinstance(obs, dict):
                obs = obs.get('obs', obs)

            # Convert to tensor if needed
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).to(self._device)

            # Capture frame from current state (renders env 0)
            frame = self._capture_frame()
            if frame is not None:
                frames.append(frame)

            # Check if first env's episode is done
            if isinstance(done, torch.Tensor):
                if done[0].item() != 0:  # DoneFlags.NULL = 0
                    break
            elif isinstance(done, np.ndarray):
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

    def _save_video_locally(self, video, iteration, prefix):
        """
        Save video to local file as MP4.

        Args:
            video: numpy array (T, H, W, C)
            iteration: Current training iteration
            prefix: Prefix for filename

        Returns:
            Path to saved file, or None if failed
        """
        if self._save_dir is None:
            return None

        try:
            import imageio
        except ImportError:
            print("[HeadlessVideoRecorder] imageio not installed, cannot save locally. Install with: pip install imageio imageio-ffmpeg")
            return None

        filename = f"{prefix}_iter{iteration:08d}.mp4"
        filepath = os.path.join(self._save_dir, filename)

        try:
            # imageio expects (T, H, W, C) uint8
            if video.dtype != np.uint8:
                video = (video * 255).astype(np.uint8)

            imageio.mimwrite(filepath, video, fps=self._fps, codec='libx264')
            return filepath
        except Exception as e:
            print(f"[HeadlessVideoRecorder] Failed to save video locally: {e}")
            return None

    def record_and_log(self, iteration, num_episodes=1, max_steps=300, prefix="policy"):
        """
        Record episodes and log video to wandb and/or save locally.

        Args:
            iteration: Current training iteration (for labeling)
            num_episodes: Number of episodes to record
            max_steps: Maximum steps per episode
            prefix: Prefix for wandb log key and local filename
        """
        # Check if we can log anywhere
        can_wandb = WANDB_AVAILABLE and wandb.run is not None
        can_local = self._save_dir is not None

        if not can_wandb and not can_local:
            print("[HeadlessVideoRecorder] No output configured (no wandb run and no save_dir)")
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

        if video is None:
            return

        saved_path = None

        # Save locally first
        if can_local:
            saved_path = self._save_video_locally(video, iteration, prefix)
            if saved_path:
                print(f"[HeadlessVideoRecorder] Saved video to {saved_path}")

        # Log to wandb
        if can_wandb:
            wandb.log({
                f"{prefix}_video": wandb.Video(
                    video,
                    fps=self._fps,
                    format="mp4"
                )
            }, step=iteration)
            print(f"[HeadlessVideoRecorder] Logged video to wandb at iteration {iteration} "
                  f"({len(all_frames)} frames, {video.shape})")


# Backwards compatibility alias
VideoRecorder = HeadlessVideoRecorder
SimpleVideoRecorder = HeadlessVideoRecorder
