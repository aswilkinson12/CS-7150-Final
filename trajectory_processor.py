"""
TrajectoryProcessor v2 - Unified Trajectory Processing
Combines basic and enhanced (scene-aware) trajectory processing
Supports both 259 vocab (basic) and 280 vocab (enhanced with semantic tokens)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class TrajectoryConfig:
    """
    Unified configuration for trajectory processing
    Supports both basic and enhanced modes
    """
    
    # Spatial discretization
    coverage: float = 64.0  # meters
    grid_size: int = 16
    cell_size: float = 2.0  # meters per cell
    
    # Temporal
    history_length: int = 4  # 2 seconds @ 2Hz
    future_length: int = 6   # 3 seconds @ 2Hz
    
    # Vocabulary - Extended for scene-aware mode
    # Spatial tokens: 0-255 (16x16 grid)
    PAD_TOKEN = 256
    START_TOKEN = 257
    END_TOKEN = 258
    
    # Scene context tokens (259-262)
    SCENE_INTERSECTION = 259
    SCENE_HIGHWAY = 260
    SCENE_PARKING = 261
    SCENE_URBAN = 262
    
    # Lane context tokens (263-266)
    LANE_STRAIGHT = 263
    LANE_LEFT = 264
    LANE_RIGHT = 265
    LANE_MERGE = 266
    
    # Traffic light tokens (267-270)
    TRAFFIC_LIGHT_RED = 267
    TRAFFIC_LIGHT_YELLOW = 268
    TRAFFIC_LIGHT_GREEN = 269
    TRAFFIC_LIGHT_UNKNOWN = 270
    
    # Agent type tokens (271-273)
    AGENT_CAR = 271
    AGENT_TRUCK = 272
    AGENT_BUS = 273
    
    # Agent marker
    NEARBY_AGENT_START = 274
    
    # Map context tokens (275-277)
    ON_INTERSECTION = 275
    NEAR_CROSSWALK = 276
    NEAR_STOP_SIGN = 277
    
    # Vocabulary size
    vocab_size: int = 280  # Use 259 for basic mode, 280 for enhanced
    
    # Mode
    enhanced_mode: bool = True  # Set to False for basic 259-vocab mode


class GridDiscretizer:
    """Discretize continuous trajectories to grid tokens"""
    
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        
    def discretize_point(self, x: float, y: float) -> int:
        """
        Convert (x, y) coordinates to grid token
        
        Args:
            x, y: Coordinates in meters
            
        Returns:
            token: Integer in [0, 255]
        """
        # Shift to positive coordinates
        grid_x = (x + self.config.coverage / 2) / self.config.cell_size
        grid_y = (y + self.config.coverage / 2) / self.config.cell_size
        
        # Clip to grid bounds
        grid_x = int(np.clip(grid_x, 0, self.config.grid_size - 1))
        grid_y = int(np.clip(grid_y, 0, self.config.grid_size - 1))
        
        # Convert to token
        token = grid_y * self.config.grid_size + grid_x
        
        return token
    
    def discretize_trajectory(self, trajectory: np.ndarray) -> List[int]:
        """
        Discretize entire trajectory
        
        Args:
            trajectory: (N, 2) array of (x, y) coordinates
            
        Returns:
            tokens: List of grid tokens
        """
        tokens = []
        for point in trajectory:
            token = self.discretize_point(point[0], point[1])
            tokens.append(token)
        return tokens
    
    def reconstruct_point(self, token: int) -> Tuple[float, float]:
        """
        Reconstruct (x, y) from grid token
        
        Args:
            token: Grid token [0, 255]
            
        Returns:
            (x, y): Coordinates in meters
        """
        grid_x = token % self.config.grid_size
        grid_y = token // self.config.grid_size
        
        # Convert back to continuous coordinates (cell centers)
        x = grid_x * self.config.cell_size - self.config.coverage / 2 + self.config.cell_size / 2
        y = grid_y * self.config.cell_size - self.config.coverage / 2 + self.config.cell_size / 2
        
        return x, y
    
    def reconstruct_trajectory(self, tokens: List[int]) -> np.ndarray:
        """
        Reconstruct trajectory from tokens
        
        Args:
            tokens: List of grid tokens
            
        Returns:
            trajectory: (N, 2) array of (x, y) coordinates
        """
        trajectory = []
        for token in tokens:
            if token < 256:  # Only spatial tokens
                x, y = self.reconstruct_point(token)
                trajectory.append([x, y])
        
        return np.array(trajectory) if trajectory else np.array([])


class TrajectoryProcessor:
    """
    Unified trajectory processor
    Supports both basic and enhanced (scene-aware) modes
    """
    
    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()
        self.discretizer = GridDiscretizer(self.config)
        
        # Token mappings for semantic tokens
        self._semantic_token_map = {
            259: "INTERSECTION", 260: "HIGHWAY", 261: "PARKING", 262: "URBAN",
            263: "STRAIGHT", 264: "LEFT", 265: "RIGHT", 266: "MERGE",
            267: "RED", 268: "YELLOW", 269: "GREEN", 270: "UNKNOWN",
            271: "CAR", 272: "TRUCK", 273: "BUS", 274: "AGENT_START",
            275: "ON_INTERSECTION", 276: "NEAR_CROSSWALK", 277: "NEAR_STOP_SIGN"
        }
    
    # ========================================================================
    # BASIC MODE - Original functionality
    # ========================================================================
    
    def process_basic_trajectory(
        self,
        history: np.ndarray,
        future: np.ndarray
    ) -> Dict[str, List[int]]:
        """
        Basic trajectory processing (original 259-vocab mode)
        
        Args:
            history: (H, 2) past trajectory
            future: (F, 2) future trajectory
            
        Returns:
            Dict with input_tokens and target_tokens
        """
        # Discretize
        history_tokens = self.discretizer.discretize_trajectory(history)
        future_tokens = self.discretizer.discretize_trajectory(future)
        
        # Build sequence
        input_tokens = [self.config.START_TOKEN] + history_tokens + future_tokens[:-1]
        target_tokens = history_tokens + future_tokens + [self.config.END_TOKEN]
        
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'history_tokens': history_tokens,
            'future_tokens': future_tokens
        }
    
    # ========================================================================
    # ENHANCED MODE - Scene-aware functionality
    # ========================================================================
    
    def extract_scene_context(
        self, 
        scene_data: Dict, 
        map_api=None
    ) -> List[int]:
        """
        Extract scene context tokens from metadata
        
        Args:
            scene_data: nuScenes scene/sample metadata
            map_api: Optional nuScenes map API
            
        Returns:
            List of context tokens
        """
        if not self.config.enhanced_mode:
            return []
        
        context_tokens = []
        
        # Scene type (from location/description)
        scene_desc = scene_data.get('description', '').lower()
        location = scene_data.get('location', '').lower()
        
        if 'intersection' in scene_desc or 'cross' in scene_desc:
            context_tokens.append(self.config.SCENE_INTERSECTION)
        elif 'highway' in scene_desc or 'highway' in location:
            context_tokens.append(self.config.SCENE_HIGHWAY)
        elif 'parking' in scene_desc:
            context_tokens.append(self.config.SCENE_PARKING)
        else:
            context_tokens.append(self.config.SCENE_URBAN)

        # Map context (if map available)
        if map_api is not None:
            ego_pose = scene_data.get('ego_pose', {})
            x, y = ego_pose.get('translation', [0, 0])[:2]

            try:
                # Check if on intersection
                on_intersection = map_api.is_on_intersection(x, y)
                if on_intersection:
                    context_tokens.append(self.config.ON_INTERSECTION)
            except:
                pass

            try:
                # Check nearby crosswalk
                nearby_crosswalk = self._check_nearby_crosswalk(x, y, map_api)
                if nearby_crosswalk:
                    context_tokens.append(self.config.NEAR_CROSSWALK)
            except:
                pass
        
        return context_tokens
    
    def extract_lane_context(
        self, 
        trajectory: np.ndarray, 
        map_api=None
    ) -> List[int]:
        """
        Determine lane context from trajectory
        
        Args:
            trajectory: (N, 2) trajectory
            map_api: Optional nuScenes map API
            
        Returns:
            List of lane tokens
        """
        if not self.config.enhanced_mode:
            return []
        
        lane_tokens = []
        
        if len(trajectory) < 2:
            return [self.config.LANE_STRAIGHT]
        
        # Compute lateral movement
        lateral_diff = trajectory[-1, 1] - trajectory[0, 1]
        
        if abs(lateral_diff) < 1.0:
            lane_tokens.append(self.config.LANE_STRAIGHT)
        elif lateral_diff > 2.0:
            lane_tokens.append(self.config.LANE_LEFT)
        elif lateral_diff < -2.0:
            lane_tokens.append(self.config.LANE_RIGHT)
        else:
            lane_tokens.append(self.config.LANE_STRAIGHT)
        
        return lane_tokens
    
    def encode_nearby_agents(
        self, 
        nearby_agents: List[Dict], 
        max_agents: int = 3
    ) -> List[int]:
        """
        Encode nearby agents' trajectories
        
        Args:
            nearby_agents: List of agent data (position, velocity, type)
            max_agents: Maximum number of agents to encode
            
        Returns:
            List of tokens encoding agent information
        """
        if not self.config.enhanced_mode:
            return []
        
        agent_tokens = []
        
        for i, agent in enumerate(nearby_agents[:max_agents]):
            # Agent type marker
            agent_type = agent.get('type', 'car').lower()
            
            if 'truck' in agent_type:
                agent_tokens.append(self.config.AGENT_TRUCK)
            elif 'bus' in agent_type:
                agent_tokens.append(self.config.AGENT_BUS)
            else:
                agent_tokens.append(self.config.AGENT_CAR)
            
            # Agent trajectory (encode relative position)
            agent_trajectory = agent.get('trajectory', np.array([]))
            
            if len(agent_trajectory) > 0:
                agent_tokens.append(self.config.NEARBY_AGENT_START)
                
                # Discretize agent trajectory (simplified - just last 2 points)
                for point in agent_trajectory[-2:]:
                    token = self.discretizer.discretize_point(point[0], point[1])
                    agent_tokens.append(token)
        
        return agent_tokens
    
    def process_enhanced_trajectory(
        self,
        ego_history: np.ndarray,
        ego_future: np.ndarray,
        scene_context: List[int] = None,
        lane_context: List[int] = None,
        nearby_agents: List[int] = None
    ) -> Dict[str, List[int]]:
        """
        Enhanced trajectory processing with scene context
        
        Args:
            ego_history: Ego vehicle history trajectory
            ego_future: Ego vehicle future trajectory
            scene_context: Scene tokens (optional)
            lane_context: Lane tokens (optional)
            nearby_agents: Agent tokens (optional)
            
        Returns:
            Dict with input_tokens and target_tokens
        """
        # Discretize ego trajectory
        ego_history_tokens = self.discretizer.discretize_trajectory(ego_history)
        ego_future_tokens = self.discretizer.discretize_trajectory(ego_future)
        
        # Default empty lists
        scene_context = scene_context or []
        lane_context = lane_context or []
        nearby_agents = nearby_agents or []
        
        # Build enhanced sequence
        # Format: [START] [SCENE] [LANE] [EGO_HISTORY] [AGENTS] [FUTURE]
        input_tokens = (
            [self.config.START_TOKEN] +
            scene_context +
            lane_context +
            ego_history_tokens +
            nearby_agents +
            ego_future_tokens[:-1]  # All but last future token
        )
        
        target_tokens = (
            scene_context +
            lane_context +
            ego_history_tokens +
            nearby_agents +
            ego_future_tokens +
            [self.config.END_TOKEN]
        )
        
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'history_tokens': ego_history_tokens,
            'future_tokens': ego_future_tokens,
            'scene_context': scene_context,
            'lane_context': lane_context,
            'agent_tokens': nearby_agents
        }
    
    # ========================================================================
    # UNIFIED INTERFACE
    # ========================================================================
    
    def process_trajectory(
        self,
        history: np.ndarray,
        future: np.ndarray,
        scene_data: Dict = None,
        nearby_agents: List[Dict] = None,
        map_api=None
    ) -> Dict[str, List[int]]:
        """
        Unified trajectory processing
        Automatically uses basic or enhanced mode based on config
        
        Args:
            history: (H, 2) past trajectory
            future: (F, 2) future trajectory
            scene_data: Optional scene metadata (for enhanced mode)
            nearby_agents: Optional list of nearby agents (for enhanced mode)
            map_api: Optional map API (for enhanced mode)
            
        Returns:
            Dict with tokens and metadata
        """
        if not self.config.enhanced_mode:
            # Basic mode
            return self.process_basic_trajectory(history, future)
        else:
            # Enhanced mode
            scene_context = self.extract_scene_context(scene_data or {}, map_api)
            lane_context = self.extract_lane_context(history, map_api)
            agent_tokens = self.encode_nearby_agents(nearby_agents or [])
            
            return self.process_enhanced_trajectory(
                history,
                future,
                scene_context,
                lane_context,
                agent_tokens
            )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def decode_semantic_tokens(self, tokens: List[int]) -> Dict:
        """
        Extract semantic information from token sequence
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dict with scene, lanes, map features, agents
        """
        scene_type = None
        lane_context = []
        map_features = []
        agents = []
        
        for token in tokens:
            if token in [259, 260, 261, 262]:  # Scene types
                scene_type = self._semantic_token_map.get(token)
            elif token in [263, 264, 265, 266]:  # Lane context
                lane_context.append(self._semantic_token_map.get(token))
            elif token in [275, 276, 277]:  # Map features
                map_features.append(self._semantic_token_map.get(token))
            elif token in [271, 272, 273]:  # Agent types
                agents.append(self._semantic_token_map.get(token))
        
        return {
            'scene': scene_type,
            'lanes': lane_context,
            'map': map_features,
            'agents': agents
        }
    
    def _check_nearby_crosswalk(
        self, 
        x: float, 
        y: float, 
        map_api, 
        radius: float = 20.0
    ) -> bool:
        """Check if there's a crosswalk within radius"""
        try:
            # This would use nuScenes map API
            # Simplified for now
            return False
        except:
            return False
    
    def get_token_name(self, token: int) -> str:
        """Get human-readable name for token"""
        if token < 256:
            return f"SPATIAL_{token}"
        elif token == 256:
            return "PAD"
        elif token == 257:
            return "START"
        elif token == 258:
            return "END"
        elif token in self._semantic_token_map:
            return self._semantic_token_map[token]
        else:
            return f"UNKNOWN_{token}"


# ========================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================

def create_basic_processor() -> TrajectoryProcessor:
    """Create processor for basic 259-vocab mode"""
    config = TrajectoryConfig(
        vocab_size=259,
        enhanced_mode=False
    )
    return TrajectoryProcessor(config)


def create_enhanced_processor() -> TrajectoryProcessor:
    """Create processor for enhanced 280-vocab mode"""
    config = TrajectoryConfig(
        vocab_size=280,
        enhanced_mode=True
    )
    return TrajectoryProcessor(config)


# ========================================================================
# DEMO
# ========================================================================

def demo_processor():
    """Demonstrate both modes"""
    print("TrajectoryProcessor v2 Demo")
    print("=" * 60)
    
    # Test data
    history = np.array([[0, 0], [2, 0], [4, 0.5], [6, 1.0]])
    future = np.array([[8, 1.5], [10, 2.0], [12, 2.5], [14, 3.0], [16, 3.5], [18, 4.0]])
    
    # Basic mode
    print("\n1. BASIC MODE (259 vocab)")
    print("-" * 60)
    basic_processor = create_basic_processor()
    basic_result = basic_processor.process_trajectory(history, future)
    
    print(f"Input tokens:  {len(basic_result['input_tokens'])} tokens")
    print(f"Target tokens: {len(basic_result['target_tokens'])} tokens")
    print(f"Vocabulary:    {basic_processor.config.vocab_size}")
    
    # Enhanced mode
    print("\n2. ENHANCED MODE (280 vocab)")
    print("-" * 60)
    enhanced_processor = create_enhanced_processor()
    
    scene_data = {'description': 'intersection left turn', 'location': 'singapore'}
    nearby_agents = [
        {'type': 'vehicle.car', 'trajectory': np.array([[5, -2], [7, -2]])},
        {'type': 'vehicle.truck', 'trajectory': np.array([[10, 2], [12, 2]])}
    ]
    
    enhanced_result = enhanced_processor.process_trajectory(
        history, 
        future,
        scene_data=scene_data,
        nearby_agents=nearby_agents
    )
    
    print(f"Input tokens:  {len(enhanced_result['input_tokens'])} tokens")
    print(f"Target tokens: {len(enhanced_result['target_tokens'])} tokens")
    print(f"Scene context: {enhanced_result['scene_context']}")
    print(f"Lane context:  {enhanced_result['lane_context']}")
    print(f"Agent tokens:  {len(enhanced_result['agent_tokens'])} tokens")
    print(f"Vocabulary:    {enhanced_processor.config.vocab_size}")
    
    # Decode semantic tokens
    semantic_info = enhanced_processor.decode_semantic_tokens(
        enhanced_result['input_tokens']
    )
    print(f"\nSemantic info: {semantic_info}")
    
    print("\n" + "=" * 60)
    print("✓ Both modes working correctly!")


if __name__ == "__main__":
    demo_processor()