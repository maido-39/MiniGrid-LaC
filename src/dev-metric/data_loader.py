"""
Data loader for experiment_log.json files.

Extracts trajectory data (step, agent_pos) from experiment logs.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


def load_experiment_log(json_path: Path) -> List[Dict]:
    """
    Load experiment_log.json file.
    
    Args:
        json_path: Path to experiment_log.json file
        
    Returns:
        List of step data dictionaries
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_trajectory(data: List[Dict]) -> np.ndarray:
    """
    Extract trajectory from experiment log data.
    
    Args:
        data: List of step data from experiment_log.json
        
    Returns:
        Numpy array of shape (N, 2) where each row is [x, y] position
    """
    trajectory = []
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        state = step_data.get('state', {})
        agent_pos = state.get('agent_pos', [None, None])
        
        if agent_pos[0] is not None and agent_pos[1] is not None:
            trajectory.append([float(agent_pos[0]), float(agent_pos[1])])
    
    return np.array(trajectory) if trajectory else np.array([]).reshape(0, 2)


def extract_trajectory_with_steps(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract trajectory with step numbers.
    
    Args:
        data: List of step data from experiment_log.json
        
    Returns:
        Tuple of (trajectory, steps) where:
        - trajectory: Numpy array of shape (N, 2) with [x, y] positions
        - steps: Numpy array of shape (N,) with step numbers
    """
    trajectory = []
    steps = []
    
    for step_data in data:
        step_num = step_data.get('step')
        if step_num is None:
            continue
        
        state = step_data.get('state', {})
        agent_pos = state.get('agent_pos', [None, None])
        
        if agent_pos[0] is not None and agent_pos[1] is not None:
            trajectory.append([float(agent_pos[0]), float(agent_pos[1])])
            steps.append(step_num)
    
    return (np.array(trajectory) if trajectory else np.array([]).reshape(0, 2),
            np.array(steps) if steps else np.array([]))


def load_all_episodes(logs_dir: Path) -> Dict[str, Dict]:
    """
    Load all episodes from logs_good directory.
    
    Args:
        logs_dir: Path to logs_good directory
        
    Returns:
        Dictionary mapping episode names to trajectory data:
        {
            'episode_name': {
                'trajectory': np.ndarray,
                'steps': np.ndarray,
                'path': Path,
                'group': str  # 'hogun', 'hogun_0125', or 'other'
            }
        }
    """
    episodes = {}
    
    # Find all experiment_log.json files
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        print(f"Warning: Logs directory does not exist: {logs_dir}")
        return episodes
    
    json_files = list(logs_dir.rglob('experiment_log.json'))
    print(f"Found {len(json_files)} experiment_log.json files")
    
    for json_path in json_files:
        # Determine group first (needed for unique episode name)
        if 'hogun_0125' in str(json_path):
            group = 'hogun_0125'
        elif 'hogun' in str(json_path) or 'Hogun' in str(json_path):
            group = 'hogun'
        else:
            group = 'other'
        
        # Extract episode name from path
        # e.g., logs_good/Episode_1_1/experiment_log.json -> Episode_1_1
        parts = json_path.parts
        episode_name = None
        
        # Find the episode folder name (parent of experiment_log.json)
        # If experiment_log.json is directly in a folder, use that folder name
        if json_path.name == 'experiment_log.json':
            episode_name = json_path.parent.name
        
        if episode_name is None or episode_name == 'logs_good' or episode_name == 'src':
            # Fallback: use parent directory name
            episode_name = json_path.parent.name
        
        # Make episode name unique by including group prefix if needed
        # This prevents conflicts between hogun/episode1 and hogun_0125/episode1
        if group in ['hogun', 'hogun_0125'] and episode_name in ['episode1', 'episode2', 'episode3']:
            episode_name = f"{group}_{episode_name}"
        
        try:
            data = load_experiment_log(json_path)
            trajectory, steps = extract_trajectory_with_steps(data)
            
            if len(trajectory) > 0:
                episodes[episode_name] = {
                    'trajectory': trajectory,
                    'steps': steps,
                    'path': json_path,
                    'group': group
                }
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
            continue
    
    return episodes


def extract_episode_number(episode_name: str) -> Optional[int]:
    """
    Extract episode number from episode name.
    
    Episode_a_b means Episode a's b-th attempt, so episode number is a.
    
    Examples:
        "Episode_1_1" -> 1
        "Episode_2_1_Test_Entropy" -> 2
        "episode1" -> 1
        "episode2" -> 2
        "Episode_3_2" -> 3
    
    Args:
        episode_name: Name of episode
        
    Returns:
        Episode number (int) or None if not found
    """
    # Match pattern: Episode_<number> or episode<number>
    match = re.search(r'[Ee]pisode[_\s]*(\d+)', episode_name)
    if match:
        return int(match.group(1))
    return None


def get_reference_path(episodes: Dict[str, Dict], reference_name: str = 'Episode_1_1') -> Optional[np.ndarray]:
    """
    Get reference trajectory by episode name.
    
    Args:
        episodes: Dictionary of all episodes
        reference_name: Name of episode to use as reference
        
    Returns:
        Reference trajectory as numpy array, or None if not found
    """
    if reference_name in episodes:
        return episodes[reference_name]['trajectory']
    
    # Try to find similar name
    for name in episodes.keys():
        if reference_name.lower() in name.lower() or name.lower() in reference_name.lower():
            return episodes[name]['trajectory']
    
    return None
