"""Command-line interface for CircleDetection package."""

import click
import cv2
import os
import csv
from pathlib import Path

from .tracking import extract_trajectory
from .smoothing import smooth_trajectory, calculate_smoothness
from .utils import load_trajectory_csv, save_trajectory_csv


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """CircleDetection - Microparticle tracking for dark field microscopy."""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--diameter', default=48, help='Expected particle diameter in pixels')
@click.option('--kalman/--no-kalman', default=True, help='Use Kalman filtering')
@click.option('--output-dir', type=click.Path(), help='Output directory (default: ./results)')
@click.option('--save-video/--no-save-video', default=True, help='Save annotated video')
@click.option('--smooth', type=click.Choice(['none', 'savgol', 'spline', 'moving']),
              default='none', help='Apply smoothing after extraction')
@click.option('--smooth-window', default=15, help='Smoothing window size')
def process(video_path, diameter, kalman, output_dir, save_video, smooth, smooth_window):
    """
    Process a video: extract trajectory with optional smoothing.

    This is the main command for processing MP4 videos. It will:
    1. Extract particle trajectory
    2. Apply optional smoothing
    3. Save trajectory CSV and annotated video

    Example: circledetect process video.mp4 --smooth savgol
    """
    click.echo(f"Processing video: {video_path}")

    # Set up output directory
    if output_dir is None:
        output_dir = Path('./results')
    else:
        output_dir = Path(output_dir)

    csv_dir = output_dir / 'csv'
    video_dir = output_dir / 'videos'
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Get base filename
    video_name = Path(video_path).stem

    # Prepare output paths
    csv_base = csv_dir / f"{video_name}_trajectory"
    if kalman:
        csv_base = Path(str(csv_base) + "_kalman")

    output_csv = str(csv_base) + ".csv"
    output_video = str(video_dir / f"{video_name}_annotated.mp4") if save_video else None

    # Extract trajectory
    click.echo(f"\nExtracting trajectory (Kalman: {kalman})...")

    def progress(frame, total):
        if frame % 30 == 0:
            click.echo(f"  Frame {frame}/{total}")

    raw_traj, filtered_traj = extract_trajectory(
        video_path,
        expected_diameter=diameter,
        use_kalman=kalman,
        save_video=save_video,
        output_video_path=output_video,
        progress_callback=progress
    )

    # Convert to dict format
    trajectory = [
        {'frame': f, 'x': x, 'y': y, 'radius': r}
        for f, x, y, r in filtered_traj
    ]

    click.echo(f"\nDetected particle in {len(trajectory)} frames")

    # Calculate smoothness
    smoothness_before = calculate_smoothness(trajectory)
    click.echo(f"Smoothness: {smoothness_before:.3f}")

    # Apply smoothing if requested
    if smooth != 'none':
        click.echo(f"\nApplying {smooth} smoothing (window: {smooth_window})...")
        trajectory = smooth_trajectory(
            trajectory,
            method=smooth,
            window_length=smooth_window
        )

        smoothness_after = calculate_smoothness(trajectory)
        improvement = smoothness_before / smoothness_after if smoothness_after > 0 else 1.0
        click.echo(f"Smoothed: {smoothness_after:.3f} ({improvement:.2f}x better)")

        # Update output path
        output_csv = str(csv_base) + f"_smoothed_{smooth}.csv"

    # Save trajectory
    save_trajectory_csv(trajectory, output_csv)
    click.echo(f"\nTrajectory saved: {output_csv}")

    if save_video:
        click.echo(f"Annotated video saved: {output_video}")

    click.echo("\n✓ Processing complete!")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.argument('trajectory_csv', type=click.Path(exists=True))
@click.option('--output-dir', type=click.Path(), help='Output directory (default: ./results)')
@click.option('--no-rotation', is_flag=True, help='Disable rotation (only center particle)')
@click.option('--annotate', is_flag=True, help='Create annotated video with axes')
@click.option('--smooth-window', default=5, help='Velocity smoothing window')
def transform(video_path, trajectory_csv, output_dir, no_rotation, annotate, smooth_window):
    """
    Transform video to particle reference frame.

    Centers the particle and optionally rotates to align velocity.

    Example: circledetect transform video.mp4 trajectory.csv --no-rotation
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from transform_to_particle_frame import transform_video_to_particle_frame
    from config import get_csv_path, get_video_path

    click.echo(f"Transforming video: {video_path}")
    click.echo(f"Using trajectory: {trajectory_csv}")

    if no_rotation:
        click.echo("Mode: Center only (no rotation)")
    else:
        click.echo("Mode: Center + rotate to align velocity")

    # Call transformation function
    transform_video_to_particle_frame(
        video_path,
        trajectory_csv,
        velocity_smoothing_window=smooth_window,
        annotate=annotate,
        apply_rotation=not no_rotation
    )

    click.echo("\n✓ Transformation complete!")


@cli.command()
@click.argument('trajectory_csv', type=click.Path(exists=True))
@click.option('--method', type=click.Choice(['savgol', 'spline', 'moving']),
              default='savgol', help='Smoothing method')
@click.option('--window', default=15, help='Window size for savgol/moving')
@click.option('--output', type=click.Path(), help='Output CSV path')
def smooth(trajectory_csv, method, window, output):
    """
    Smooth an existing trajectory CSV.

    Example: circledetect smooth trajectory.csv --method savgol --window 15
    """
    click.echo(f"Loading trajectory: {trajectory_csv}")

    # Load trajectory
    trajectory = load_trajectory_csv(trajectory_csv)
    click.echo(f"Loaded {len(trajectory)} points")

    # Calculate original smoothness
    smoothness_before = calculate_smoothness(trajectory)
    click.echo(f"Original smoothness: {smoothness_before:.3f}")

    # Apply smoothing
    click.echo(f"\nApplying {method} smoothing...")
    smoothed = smooth_trajectory(
        trajectory,
        method=method,
        window_length=window
    )

    smoothness_after = calculate_smoothness(smoothed)
    improvement = smoothness_before / smoothness_after if smoothness_after > 0 else 1.0
    click.echo(f"Smoothed: {smoothness_after:.3f} ({improvement:.2f}x better)")

    # Determine output path
    if output is None:
        base = Path(trajectory_csv).stem
        output = Path(trajectory_csv).parent / f"{base}_smoothed_{method}.csv"

    # Save
    save_trajectory_csv(smoothed, str(output))
    click.echo(f"\n✓ Saved: {output}")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--diameter', default=48, help='Expected particle diameter')
@click.option('--smooth', type=click.Choice(['none', 'savgol', 'spline']),
              default='savgol', help='Smoothing method')
@click.option('--smooth-window', default=15, help='Smoothing window')
@click.option('--no-rotation', is_flag=True, help='Disable rotation in transform')
@click.option('--output-dir', type=click.Path(), help='Output directory')
def auto(video_path, diameter, smooth, smooth_window, no_rotation, output_dir):
    """
    Full automatic pipeline: extract → smooth → transform.

    This is the easiest way to process a video with recommended settings.

    Example: circledetect auto video.mp4 --no-rotation
    """
    click.echo("=" * 70)
    click.echo("CIRCLEDETECTION - AUTOMATIC PIPELINE")
    click.echo("=" * 70)

    # Set up output directory
    if output_dir is None:
        output_dir = Path('./results')
    else:
        output_dir = Path(output_dir)

    csv_dir = output_dir / 'csv'
    video_dir = output_dir / 'videos'
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem

    # Step 1: Extract trajectory
    click.echo("\n[1/3] Extracting trajectory with Kalman filtering...")

    def progress(frame, total):
        if frame % 30 == 0:
            click.echo(f"      Frame {frame}/{total}")

    raw_traj, filtered_traj = extract_trajectory(
        video_path,
        expected_diameter=diameter,
        use_kalman=True,
        save_video=True,
        output_video_path=str(video_dir / f"{video_name}_tracking.mp4"),
        progress_callback=progress
    )

    trajectory = [
        {'frame': f, 'x': x, 'y': y, 'radius': r}
        for f, x, y, r in filtered_traj
    ]

    click.echo(f"      ✓ Detected in {len(trajectory)} frames")

    # Save initial trajectory
    traj_csv = csv_dir / f"{video_name}_trajectory_kalman.csv"
    save_trajectory_csv(trajectory, str(traj_csv))

    # Step 2: Smooth trajectory
    if smooth != 'none':
        click.echo(f"\n[2/3] Smoothing trajectory ({smooth}, window={smooth_window})...")

        smoothness_before = calculate_smoothness(trajectory)
        trajectory = smooth_trajectory(
            trajectory,
            method=smooth,
            window_length=smooth_window
        )
        smoothness_after = calculate_smoothness(trajectory)
        improvement = smoothness_before / smoothness_after if smoothness_after > 0 else 1.0

        click.echo(f"      ✓ Smoothness improved {improvement:.2f}x")

        traj_csv = csv_dir / f"{video_name}_trajectory_smoothed.csv"
        save_trajectory_csv(trajectory, str(traj_csv))
    else:
        click.echo("\n[2/3] Skipping smoothing (--smooth none)")

    # Step 3: Transform video
    click.echo(f"\n[3/3] Transforming to particle frame (rotation: {not no_rotation})...")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from transform_to_particle_frame import transform_video_to_particle_frame

    transform_video_to_particle_frame(
        video_path,
        str(traj_csv),
        velocity_smoothing_window=5,
        annotate=True,
        apply_rotation=not no_rotation
    )

    click.echo("\n" + "=" * 70)
    click.echo("✓ PIPELINE COMPLETE!")
    click.echo("=" * 70)
    click.echo(f"\nOutputs in: {output_dir}/")
    click.echo("  - Trajectory CSVs in csv/")
    click.echo("  - Videos in videos/")


if __name__ == '__main__':
    cli()
