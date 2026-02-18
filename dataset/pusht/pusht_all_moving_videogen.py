"""
Generate videos of moving objects (circle and T-shapes) with specular reflection.
No agent or policy - just physics-based motion with wall reflections.
Matches the object sizes and resolution from stable-worldmodel/stable_worldmodel/envs/pusht/env.py
"""

import argparse
import cv2
import numpy as np
import os
import subprocess
import tempfile
import pygame
import pymunk
import pymunk.pygame_util
from pathlib import Path

# Disable pygame audio to avoid ALSA errors in headless environments
os.environ['SDL_AUDIODRIVER'] = 'dummy'


class MovingObjectsVideoGenerator:
    """
    Generates videos with moving objects matching PushT environment specifications.

    Object sizes (from env.py):
    - Circle (agent): radius = 0.375 * scale, where scale=40 -> radius=15
    - Tee: scale=30, length=4 (hardcoded in env.py add_tee)

    Resolution:
    - window_size: 512 (physics simulation)
    - render_size: 224 (output video)
    """

    # Constants matching env.py
    WINDOW_SIZE = 512
    RENDER_SIZE = 224

    # Circle parameters (from env.py add_circle)
    CIRCLE_BASE_RADIUS = 0.375
    CIRCLE_SCALE = 40
    CIRCLE_RADIUS = int(CIRCLE_BASE_RADIUS * CIRCLE_SCALE)  # = 15

    # Tee parameters (from env.py add_tee - note: scale is hardcoded to 30 in add_tee)
    TEE_SCALE = 30
    TEE_LENGTH = 4

    def __init__(
        self,
        fps=40,
        seed=None,
    ):
        self.window_size = self.WINDOW_SIZE
        self.render_size = self.RENDER_SIZE
        self.fps = fps
        self.rng = np.random.default_rng(seed)

        # Initialize pygame (without display for headless)
        pygame.init()
        self.screen = pygame.Surface((self.window_size, self.window_size))

        # Physics setup
        self.space = None
        self.circle_body = None
        self.circle_shape = None
        self.main_T_body = None
        self.main_T_shapes = None
        self.sub_T_body = None
        self.sub_T_shapes = None

    def setup_space(self):
        """Create pymunk space with walls."""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity
        # CRITICAL: damping=1.0 means velocity is fully preserved (no decay)
        # damping=0.0 would mean velocity immediately becomes 0!
        self.space.damping = 1.0

        # Add walls matching env.py (line 481-485)
        walls = [
            self._add_wall((5, 506), (5, 5)),      # Left wall
            self._add_wall((5, 5), (506, 5)),      # Top wall
            self._add_wall((506, 5), (506, 506)),  # Right wall
            self._add_wall((5, 506), (506, 506)),  # Bottom wall
        ]
        for wall in walls:
            self.space.add(wall)

    def _add_wall(self, a, b):
        """Add a wall segment with perfect elasticity."""
        shape = pymunk.Segment(self.space.static_body, a, b, 2)
        shape.elasticity = 1.0  # Perfect elastic collision
        shape.friction = 0.0    # No friction
        shape.collision_type = 0
        return shape

    def add_circle(self, position, velocity, color="RoyalBlue", enable_collision=True):
        """
        Add a moving circle matching env.py specifications.
        radius = 0.375 * 40 = 15 (from env.py add_circle)

        Args:
            enable_collision: If True, circle collides with main_T (elastic collision).
                            If False, circle passes through all objects.
        """
        radius = self.CIRCLE_RADIUS
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, radius)

        body = pymunk.Body(mass, moment)
        body.position = position
        body.velocity = velocity

        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        shape.elasticity = 1.0
        shape.friction = 0.0

        # Categories: circle=0b001, main_T=0b010, sub_T=0b100
        if enable_collision:
            # Circle collides with main_T (mask=0b011 includes main_T category)
            shape.filter = pymunk.ShapeFilter(categories=0b001, mask=0b011)
        else:
            # Circle does not collide with any objects (mask=0b000)
            shape.filter = pymunk.ShapeFilter(categories=0b001, mask=0b000)
        shape.collision_type = 1

        self.space.add(body, shape)
        return body, shape

    def add_tee(self, position, velocity, angle=0, color="LightSlateGray",
                categories=0b010, mask=0b001, collision_type=2):
        """
        Add a T-shaped object matching env.py add_tee specifications.
        scale=30, length=4 (hardcoded in env.py)

        Vertices from env.py add_tee (lines 570-582):
        vertices1 (horizontal bar): [(-60, 30), (60, 30), (60, 0), (-60, 0)]
        vertices2 (vertical stem): [(-15, 30), (-15, 120), (15, 120), (15, 30)]
        """
        scale = self.TEE_SCALE
        length = self.TEE_LENGTH
        mass = 1.0

        # Vertices exactly matching env.py add_tee
        vertices1 = [
            (-length * scale / 2, scale),   # (-60, 30)
            (length * scale / 2, scale),    # (60, 30)
            (length * scale / 2, 0),        # (60, 0)
            (-length * scale / 2, 0),       # (-60, 0)
        ]
        vertices2 = [
            (-scale / 2, scale),            # (-15, 30)
            (-scale / 2, length * scale),   # (-15, 120)
            (scale / 2, length * scale),    # (15, 120)
            (scale / 2, scale),             # (15, 30)
        ]

        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices2)

        body = pymunk.Body(mass, inertia1 + inertia2)

        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)

        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.elasticity = 1.0
        shape2.elasticity = 1.0
        shape1.friction = 0.0
        shape2.friction = 0.0

        # Set collision filters
        shape1.filter = pymunk.ShapeFilter(categories=categories, mask=mask)
        shape2.filter = pymunk.ShapeFilter(categories=categories, mask=mask)
        shape1.collision_type = collision_type
        shape2.collision_type = collision_type

        # Set center of gravity (matching env.py)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.velocity = velocity
        # Prevent rotation
        body.moment = float('inf')

        self.space.add(body, shape1, shape2)
        return body, (shape1, shape2)

    def initialize_objects(
        self,
        min_speed=150,
        max_speed=300,
        circle_color="RoyalBlue",
        main_T_color="LightSlateGray",
        sub_T_color="LightSlateGray",
        enable_collision=True,
    ):
        """Initialize three objects with random positions and velocities.

        Args:
            enable_collision: If True, circle and main_T undergo elastic collision.
                            If False, all objects pass through each other.
        """
        margin = 120  # Keep objects away from walls initially

        def random_position():
            x = float(self.rng.uniform(margin, self.window_size - margin))
            y = float(self.rng.uniform(margin, self.window_size - margin))
            return (x, y)

        def random_velocity():
            angle = float(self.rng.uniform(0, 2 * np.pi))
            speed = float(self.rng.uniform(min_speed, max_speed))
            return (speed * np.cos(angle), speed * np.sin(angle))

        # Create circle (agent)
        circle_pos = random_position()
        circle_vel = random_velocity()
        self.circle_body, self.circle_shape = self.add_circle(
            circle_pos, circle_vel, color=circle_color, enable_collision=enable_collision
        )

        # Create main_T
        # If collision enabled: main_T collides with circle (elastic collision)
        # If collision disabled: main_T passes through everything
        main_T_pos = random_position()
        main_T_vel = random_velocity()
        main_T_angle = float(self.rng.uniform(0, 2 * np.pi))
        if enable_collision:
            # main_T categories=0b010, mask=0b001 (collides with circle)
            main_T_categories = 0b010
            main_T_mask = 0b001
        else:
            # main_T categories=0b010, mask=0b000 (collides with nothing)
            main_T_categories = 0b010
            main_T_mask = 0b000
        self.main_T_body, self.main_T_shapes = self.add_tee(
            main_T_pos, main_T_vel, angle=main_T_angle, color=main_T_color,
            categories=main_T_categories, mask=main_T_mask, collision_type=2
        )

        # Create sub_T - always does NOT collide with circle or main_T (passes through)
        # sub_T categories=0b100, mask=0b000 (collides with nothing except walls)
        sub_T_pos = random_position()
        sub_T_vel = random_velocity()
        sub_T_angle = float(self.rng.uniform(0, 2 * np.pi))
        self.sub_T_body, self.sub_T_shapes = self.add_tee(
            sub_T_pos, sub_T_vel, angle=sub_T_angle, color=sub_T_color,
            categories=0b100, mask=0b000, collision_type=3
        )

    def _draw_circle(self, surface, body, shape):
        """Draw a circle on the surface."""
        pos = int(body.position.x), int(body.position.y)
        radius = int(shape.radius)
        pygame.draw.circle(surface, shape.color, pos, radius)

    def _draw_tee(self, surface, body, shapes):
        """Draw a T-shape on the surface."""
        for shape in shapes:
            # Get the vertices in world coordinates
            vertices = [v.rotated(body.angle) + body.position for v in shape.get_vertices()]
            points = [(int(v.x), int(v.y)) for v in vertices]
            pygame.draw.polygon(surface, shape.color, points)

    def _add_video_realism(self, img, jpeg_quality=85):
        """Add JPEG compression and noise to make video look realistic.
        
        This simulates realistic video compression and sensor noise to match
        actual video data like PushT which has significant pixel variation.
        
        Args:
            img: RGB image array (H, W, 3)
            jpeg_quality: JPEG quality level (0-100, lower = more artifacts)
        
        Returns:
            Modified image with realistic video artifacts
        """
        img_uint8 = img.astype(np.uint8)
        
        # 1. Simulate JPEG compression by encoding and decoding multiple times
        # Multiple passes create more realistic block artifacts
        img_compressed = img_uint8
        for _ in range(2):  # Double-encode for stronger artifacts
            _, jpeg_data = cv2.imencode('.jpg', cv2.cvtColor(img_compressed, cv2.COLOR_RGB2BGR), 
                                         [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            img_compressed = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
            img_compressed = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)
        
        # 2. Add significant Gaussian noise to match real video grain
        # Real camera sensor noise is ~3-8 units (std dev) on pixel values
        # This creates the 10-20 unit variation we see in actual PushT data
        noise_level = self.rng.normal(0, 4.0, img_compressed.shape)  # Increased from 1.5 to 4.0
        img_noisy = np.clip(img_compressed.astype(np.float32) + noise_level, 0, 255)
        
        # 3. Add spatial correlation to noise (more realistic than pure Gaussian)
        # Slight blur of noise creates correlated pixel patterns like real video
        kernel_size = 3
        img_noisy_blur = cv2.GaussianBlur(img_noisy.astype(np.uint8), (kernel_size, kernel_size), 0.7)
        # Mix original noisy with slightly blurred version
        img_noisy = 0.6 * img_noisy + 0.4 * img_noisy_blur
        
        # 4. Add per-channel color jitter for more variation
        # Camera sensors have different noise characteristics per channel
        color_jitter = self.rng.normal(1.0, 0.015, 3)  # Increased from 0.005 to 0.015 (~1.5%)
        img_jittered = img_noisy * color_jitter[np.newaxis, np.newaxis, :]
        img_jittered = np.clip(img_jittered, 0, 255)
        
        # 5. Add small amount of salt-and-pepper-like noise for additional variation
        # This creates occasional outlier values like in real data
        noise_mask = self.rng.random(img_jittered.shape) < 0.001  # 0.1% of pixels
        salt_pepper = self.rng.choice([0, 255], size=img_jittered.shape)
        img_jittered = np.where(noise_mask, salt_pepper, img_jittered)
        
        return np.clip(img_jittered, 0, 255).astype(np.uint8)

    def render_frame(self, background_color=(255, 255, 255), add_noise=True, jpeg_quality=85):
        """Render the current state to a frame with optional noise and compression.
        
        Args:
            background_color: RGB tuple for background
            add_noise: If True, add Gaussian noise to simulate video grain
            jpeg_quality: JPEG quality (0-100) to simulate compression artifacts
        """
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(background_color)

        # Draw objects (sub_T first so it's behind, then main_T, then circle on top)
        self._draw_tee(canvas, self.sub_T_body, self.sub_T_shapes)
        self._draw_tee(canvas, self.main_T_body, self.main_T_shapes)
        self._draw_circle(canvas, self.circle_body, self.circle_shape)

        # Convert to numpy array and resize
        img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
        img = cv2.resize(img, (self.render_size, self.render_size))

        # Add noise to simulate realistic video
        if add_noise:
            img = self._add_video_realism(img, jpeg_quality=jpeg_quality)

        return img

    def _reflect_at_walls(self, body, margin):
        """Manually reflect object velocity when hitting walls."""
        x, y = body.position
        vx, vy = body.velocity

        # Wall boundaries (matching env.py walls at 5 and 506)
        min_bound = 5 + margin
        max_bound = 506 - margin

        # Reflect at boundaries
        if x < min_bound:
            body.position = (min_bound, y)
            vx = abs(vx)
        elif x > max_bound:
            body.position = (max_bound, y)
            vx = -abs(vx)

        x, y = body.position  # Update after potential x change

        if y < min_bound:
            body.position = (x, min_bound)
            vy = abs(vy)
        elif y > max_bound:
            body.position = (x, max_bound)
            vy = -abs(vy)

        body.velocity = (vx, vy)

    def step_simulation(self, dt):
        """Step the physics simulation and handle wall reflections."""
        # Step physics (this handles circle-main_T collisions via pymunk)
        self.space.step(dt)

        # Manual wall reflection for all objects
        # Circle margin = radius
        self._reflect_at_walls(self.circle_body, margin=self.CIRCLE_RADIUS)
        # Tee margin = approximate size (longest dimension is ~120 from center)
        self._reflect_at_walls(self.main_T_body, margin=70)
        self._reflect_at_walls(self.sub_T_body, margin=70)

    def generate_video(
        self,
        output_path,
        num_frames=120,
        min_speed=150,
        max_speed=300,
        background_color=(255, 255, 255),
        circle_color="RoyalBlue",
        main_T_color="LightSlateGray",
        sub_T_color="LightSlateGray",
        enable_collision=True,
        add_noise=True,
        jpeg_quality=85,
    ):
        """Generate a video of moving objects.

        Args:
            enable_collision: If True, circle and main_T undergo elastic collision.
                            If False, all objects pass through each other.
            add_noise: If True, add JPEG compression and noise to simulate realistic video.
            jpeg_quality: JPEG quality (0-100) for compression artifacts (lower = more artifacts).
        """
        # Setup physics space
        self.setup_space()

        # Initialize objects with random positions and velocities
        self.initialize_objects(
            min_speed=min_speed,
            max_speed=max_speed,
            circle_color=circle_color,
            main_T_color=main_T_color,
            sub_T_color=sub_T_color,
            enable_collision=enable_collision,
        )

        # Collect all frames
        frames = []
        dt = 1.0 / self.fps

        for frame_idx in range(num_frames):
            # Render current state
            frame = self.render_frame(
                background_color=background_color,
                add_noise=add_noise,
                jpeg_quality=jpeg_quality
            )
            frames.append(frame)

            # Step physics for next frame
            self.step_simulation(dt)

        # Save video using the best available method
        self._save_video(output_path, frames)

    def _save_video(self, output_path, frames):
        """Save frames as video using the best available method."""
        output_path = Path(output_path)

        # Try torchcodec first
        try:
            self._save_with_torchcodec(output_path, frames)
            return
        except Exception as e:
            pass

        # Try ffmpeg via subprocess
        try:
            self._save_with_ffmpeg(output_path, frames)
            return
        except Exception as e:
            pass

        # Fallback to cv2 with various codecs
        self._save_with_cv2(output_path, frames)

    def _save_with_torchcodec(self, output_path, frames):
        """Save video using torchcodec."""
        import torch
        from torchcodec.encoders import VideoEncoder

        # Stack frames into tensor (T, H, W, C) -> (T, C, H, W)
        frames_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)

        # Create encoder and encode
        encoder = VideoEncoder(
            output_path=str(output_path),
            video_codec="libx264",
            width=self.render_size,
            height=self.render_size,
            fps=self.fps,
        )
        encoder.add_frames(frames_tensor)
        encoder.finish()

    def _save_with_ffmpeg(self, output_path, frames):
        """Save video using ffmpeg subprocess."""
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save frames as images
            for i, frame in enumerate(frames):
                frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Use ffmpeg to create video
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(self.fps),
                "-i", os.path.join(tmpdir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(output_path)
            ]
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    def _save_with_cv2(self, output_path, frames):
        """Save video using cv2 with fallback codecs."""
        output_path = str(output_path)

        # Try different codecs in order of preference
        codecs = [
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
        ]

        for codec, ext in codecs:
            try:
                # Adjust output path extension if needed
                if not output_path.endswith(ext):
                    actual_path = output_path.rsplit('.', 1)[0] + ext
                else:
                    actual_path = output_path

                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(
                    actual_path,
                    fourcc,
                    self.fps,
                    (self.render_size, self.render_size)
                )

                if not out.isOpened():
                    continue

                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                out.release()

                # Verify file was created and has content
                if os.path.exists(actual_path) and os.path.getsize(actual_path) > 0:
                    # If we had to change extension, notify
                    if actual_path != output_path:
                        print(f"  Note: Saved as {actual_path} (codec {codec})")
                    return

            except Exception:
                continue

        raise RuntimeError("Could not save video with any available codec")

    def close(self):
        """Clean up pygame."""
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Generate videos of moving objects with specular reflection"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="your/path/to/data/pusht_independent_videos_with_noise",
        help="Output directory for generated videos"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=10000,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=120,
        help="Number of frames per video (120 frames at 40fps = 3 seconds)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=40,
        help="Frames per second (40fps for 120 frames = 3 seconds)"
    )
    parser.add_argument(
        "--min_speed",
        type=float,
        default=150,
        help="Minimum object speed (pixels/second)"
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=250,
        help="Maximum object speed (pixels/second)"
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=0,
        help="Base random seed (each video uses base_seed + video_index)"
    )
    parser.add_argument(
        "--circle_color",
        type=str,
        default="#618aff",
        help="Color of the circle (hex code)"
    )
    parser.add_argument(
        "--main_T_color",
        type=str,
        default="#9cadc0",
        help="Color of the main_T (hex code)"
    )
    parser.add_argument(
        "--sub_T_color",
        type=str,
        default="#a0ee99",
        help="Color of the sub_T (hex code)"
    )
    parser.add_argument(
        "--enable_collision",
        action="store_true",
        default=False,
        help="If set, circle and main_T undergo elastic collision. Otherwise all objects pass through each other."
    )
    parser.add_argument(
        "--add_noise",
        action="store_true",
        default=True,
        help="If set, add JPEG compression and noise to simulate realistic video (default: True)."
    )
    parser.add_argument(
        "--no_noise",
        action="store_true",
        default=False,
        help="If set, disable noise/compression (generates clean synthetic video)."
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=85,
        help="JPEG quality level (0-100, lower = more compression artifacts). Default 85 mimics typical video compression."
    )

    args = parser.parse_args()

    # Handle noise flags
    add_noise = not args.no_noise and args.add_noise

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_videos} videos in {output_dir}")
    print(f"Each video: {args.num_frames} frames at {args.fps} fps ({args.num_frames/args.fps:.1f} seconds)")
    print(f"Resolution: {MovingObjectsVideoGenerator.WINDOW_SIZE}x{MovingObjectsVideoGenerator.WINDOW_SIZE} -> {MovingObjectsVideoGenerator.RENDER_SIZE}x{MovingObjectsVideoGenerator.RENDER_SIZE}")
    print(f"Circle radius: {MovingObjectsVideoGenerator.CIRCLE_RADIUS}, Tee scale: {MovingObjectsVideoGenerator.TEE_SCALE}")
    print(f"Collision between circle and main_T: {'Enabled' if args.enable_collision else 'Disabled'}")
    print(f"Video realism (JPEG compression + noise): {'Enabled' if add_noise else 'Disabled'}")
    if add_noise:
        print(f"  JPEG quality: {args.jpeg_quality} (lower = more artifacts)")
    print(f"Base seed: {args.base_seed}")
    print("-" * 60)

    # Generate N videos
    for video_idx in range(args.num_videos):
        # Create unique seed for this video
        video_seed = args.base_seed + video_idx

        # Create output filename with zero-padding
        num_digits = len(str(args.num_videos))
        output_filename = f"video_{video_idx:0{num_digits}d}.mp4"
        output_path = output_dir / output_filename

        # Create generator with unique seed
        generator = MovingObjectsVideoGenerator(
            fps=args.fps,
            seed=video_seed,
        )

        # Generate video
        print(f"[{video_idx + 1}/{args.num_videos}] Generating {output_filename} (seed={video_seed})...")
        generator.generate_video(
            output_path=output_path,
            num_frames=args.num_frames,
            min_speed=args.min_speed,
            max_speed=args.max_speed,
            circle_color=args.circle_color,
            main_T_color=args.main_T_color,
            sub_T_color=args.sub_T_color,
            enable_collision=args.enable_collision,
            add_noise=add_noise,
            jpeg_quality=args.jpeg_quality,
        )

        # Clean up
        generator.close()

    print("-" * 60)
    print(f"Successfully generated {args.num_videos} videos in {output_dir}")


if __name__ == "__main__":
    main()