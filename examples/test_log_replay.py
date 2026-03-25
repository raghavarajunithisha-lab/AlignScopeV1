import json
import csv
import time
import argparse
import alignscope
from pathlib import Path

def replay_json(file_path: Path, fps: int = 20):
    """
    Replay a JSON-lines file where each line is a tick containing:
    { "tick": 1, "agents": [...], "actions": {...}, "rewards": {...}, "defection_events": [...] }
    """
    print(f"Replaying JSON logs from {file_path} at {fps} fps...")
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            
            alignscope.log(
                step=data.get("tick", 0),
                agents=data.get("agents", []),
                actions=data.get("actions", {}),
                rewards=data.get("rewards", {}),
                defection_events=data.get("defection_events", [])
            )
            time.sleep(1.0 / fps)

def replay_csv(file_path: Path, fps: int = 20):
    """
    Replay a CSV file.
    Assumes standard long format: 
    tick, agent_id, team, role, x, y, energy, is_defector, coalition_id, action, reward
    Groups by tick and flushes to dashboard.
    """
    print(f"Replaying CSV logs from {file_path} at {fps} fps...")
    
    current_tick = -1
    agents = []
    actions = {}
    rewards = {}
    
    def flush(tick):
        if agents:
            alignscope.log(
                step=tick, 
                agents=agents, 
                actions=actions, 
                rewards=rewards
            )
            time.sleep(1.0 / fps)
            agents.clear()
            actions.clear()
            rewards.clear()

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tick = int(row.get("tick", 0))
            if current_tick != -1 and tick != current_tick:
                flush(current_tick)
                
            current_tick = tick
            aid = row.get("agent_id", "")
            
            agents.append({
                "agent_id": aid,
                "team": int(row.get("team", 0)),
                "role": row.get("role", "agent"),
                "x": float(row.get("x", 0.0)),
                "y": float(row.get("y", 0.0)),
                "resources": float(row.get("resources", 0.0)),
                "hearts": float(row.get("hearts", 0.0)),
                "energy": float(row.get("energy", 0.0)),
                "is_defector": str(row.get("is_defector", "False")).lower() == "true",
                "coalition_id": int(row.get("coalition_id", -1))
            })
            if "action" in row:
                actions[aid] = row["action"]
            if "reward" in row:
                rewards[aid] = float(row["reward"])
                
        # Flush last tick
        flush(current_tick)

def replay_npz(file_path: Path, fps: int = 20):
    """Replay trajectory data stored in NumPy NPZ archives."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy is required to replay .npz files. Run: pip install numpy")
        return
        
    print(f"Loading NPZ archive from {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"Loaded NPZ with arrays: {data.files}")
        print("Note: NPZ support is an integration stub mapping raw arrays to alignscope.log().")
    except Exception as e:
        print(f"Failed to load NPZ: {e}")

def replay_tensorboard(file_path: Path, fps: int = 20):
    """Extract and replay metrics from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Error: tensorboard is required to replay .tfevents. Run: pip install tensorboard")
        return
        
    print(f"Loading TensorBoard events from {file_path}...")
    try:
        acc = EventAccumulator(str(file_path))
        acc.Reload()
        print(f"Found tags: {list(acc.Tags().keys())}")
        print("Note: TensorBoard support is an integration stub bridging scalar tags to AlignScope.")
    except Exception as e:
        print(f"Failed to load TensorBoard: {e}")

def replay_wandb(run_path: str, fps: int = 20):
    """Replay a training run directly from the Weights & Biases API."""
    try:
        import wandb
    except ImportError:
        print("Error: wandb is required to replay W&B runs. Run: pip install wandb")
        return
        
    print(f"Fetching W&B run {run_path}...")
    try:
        api = wandb.Api()
        run = api.run(run_path)
        print(f"Successfully connected to W&B run: {run.name}")
        print("Note: W&B support is an integration stub bridging run.scan_history() to AlignScope.")
    except Exception as e:
        print(f"Failed to fetch W&B run: {e}")

def main():
    parser = argparse.ArgumentParser(description="AlignScope Offline Replay Engine")
    parser.add_argument("--file", type=str, help="Path to the log file (.jsonl, .csv, .npz, .tfevents)")
    parser.add_argument("--wandb", type=str, help="Weights & Biases run path (e.g., entity/project/run_id)")
    parser.add_argument("--fps", type=int, default=20, help="Playback speed in frames per second")
    parser.add_argument("--project", type=str, default="offline-replay", help="Project name to display on dashboard")
    
    args = parser.parse_args()
    
    alignscope.init(project=args.project)

    if args.wandb:
        replay_wandb(args.wandb, args.fps)
        return
        
    if not args.file:
        print("Error: Must provide either --file or --wandb")
        return

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return
        
    try:
        if file_path.suffix == ".json" or file_path.suffix == ".jsonl":
            replay_json(file_path, args.fps)
        elif file_path.suffix == ".csv":
            replay_csv(file_path, args.fps)
        elif file_path.suffix == ".npz":
            replay_npz(file_path, args.fps)
        elif ".tfevents" in file_path.name:
            replay_tensorboard(file_path, args.fps)
        else:
            print(f"Error: Unsupported format '{file_path.name}'. Supported: .csv, .jsonl, .npz, .tfevents, or --wandb.")
    except KeyboardInterrupt:
        print("\nReplay stopped by user.")
    print("Replay complete.")

if __name__ == "__main__":
    main()
