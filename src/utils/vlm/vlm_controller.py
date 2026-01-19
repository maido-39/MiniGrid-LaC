"""
Generic controller class for controlling environments using VLM

This module provides a generic controller class for manipulating environments
using VLM (Vision Language Model). Works independently of the environment and
is compatible with various environments through Protocol-based interfaces.

Key features:
- Action generation using VLM
- Prompt management (easily manipulable at instantiation)
- Environment state visualization (environment-specific visualization uses separate helpers)
- VLM response parsing and action execution
"""

from .vlm_wrapper import ChatGPT4oVLMWrapper
from .vlm_postprocessor import VLMResponsePostProcessor
import numpy as np
from typing import Union, Tuple, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """
    Environment interface required by VLM controller
    
    All environments implementing this Protocol are compatible with VLMController.
    """
    
    def get_image(self, fov_range: Optional[int] = None, fov_width: Optional[int] = None) -> np.ndarray:
        """Return current environment image (for VLM input)"""
        ...
    
    def get_state(self) -> Dict:
        """Return current environment state information"""
        ...
    
    def step(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute action"""
        ...
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Initialize environment"""
        ...
    
    def parse_absolute_action(self, action_str: str) -> int:
        """Convert absolute direction action string to index (optional)"""
        ...
    
    def get_absolute_action_space(self) -> Dict:
        """Return absolute direction action space information (optional)"""
        ...


class VLMController:
    """
    Generic controller class for controlling environments using VLM
    
    This class is compatible with all environments implementing EnvironmentProtocol.
    Does not include environment-specific code, uses only VLM and environment interface.
    
    Usage examples:
        # Create environment (in separate file per project)
        from my_project.environment import create_my_environment
        env = create_my_environment()
        env.reset()
        
        # Create controller
        controller = VLMController(
            env=env,
            model="gpt-4o",
            system_prompt="You are a robot...",
            user_prompt_template="Complete the mission: {mission}"
        )
        
        # Generate and execute action with VLM
        response = controller.generate_action()
        controller.execute_action(response['action'])
    """
    
    def __init__(
        self,
        env: EnvironmentProtocol,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        required_fields: Optional[list] = None
    ):
        """Initialize VLM controller for environment control.
        
        Creates a controller that uses a Vision Language Model (VLM) to generate
        actions for controlling an environment. The controller captures environment
        images, sends them to the VLM with prompts, and parses the responses
        to extract actions.
        
        Args:
            env: Environment instance that implements EnvironmentProtocol.
                Must have methods: get_image(), get_state(), step(), reset(),
                and optionally parse_absolute_action() and get_absolute_action_space().
            model: VLM model name to use. Defaults to "gpt-4o".
                Other options: "gpt-4o-mini", "gpt-4-turbo", etc.
            temperature: Generation temperature for the VLM (0.0-2.0).
                Lower values (0.0-0.3) produce more deterministic responses.
                Higher values (0.7-2.0) produce more creative/varied responses.
                Defaults to 0.0 for consistent robot control.
            max_tokens: Maximum number of tokens in the VLM response.
                Defaults to 1000, which is usually sufficient for action commands.
            system_prompt: Custom system prompt that defines the VLM's role
                and behavior. If None, uses a default prompt optimized for
                absolute direction movement control. Defaults to None.
            user_prompt_template: Template string for user prompts. Should
                contain {mission} placeholder. If None, uses default template.
                Defaults to None.
            required_fields: List of field names that must be present in the
                VLM response. The postprocessor will validate these fields.
                Defaults to ["action", "environment_info"].
        
        Examples:
            >>> from utils import MiniGridEmojiWrapper, VLMController
            >>> 
            >>> # Create environment
            >>> env = MiniGridEmojiWrapper(size=10)
            >>> env.reset()
            >>> 
            >>> # Create controller with default settings
            >>> controller = VLMController(env=env)
            >>> 
            >>> # Create controller with custom prompts
            >>> controller = VLMController(
            ...     env=env,
            ...     model="gpt-4o",
            ...     system_prompt="You are a robot controller...",
            ...     user_prompt_template="Mission: {mission}. What action?"
            ... )
            >>> 
            >>> # Create controller with custom required fields
            >>> controller = VLMController(
            ...     env=env,
            ...     required_fields=["action", "reasoning", "confidence"]
            ... )
        
        Note:
            The environment must implement EnvironmentProtocol. Most MiniGrid
            wrappers (like MiniGridEmojiWrapper) are compatible out of the box.
        """
        self.env = env
        
        self.vlm = ChatGPT4oVLMWrapper(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.postprocessor = VLMResponsePostProcessor(
            required_fields=required_fields or ["action", "environment_info"]
        )
        
        # Set prompts
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.user_prompt_template = user_prompt_template or self._get_default_user_prompt_template()
    
    def _get_default_system_prompt(self) -> str:
        """Return default system prompt (based on absolute direction movement)"""
        return """You are a robot operating on a grid map.

## Environment
Grid world with walls, objects, robot, and goal markers.

## Coordinate System
The top of the image is North (up), and the bottom is South (down).
The left is West (left), and the right is East (right).

## Robot Orientation
The robot's heading direction is shown in the image.
However, you can move in ANY direction regardless of the robot's current heading.

## Action Space (Absolute Directions)
You can move directly in absolute directions:
- "up": Move North
- "down": Move South
- "left": Move West
- "right": Move East
- "pickup": Pick up object
- "drop": Drop object
- "toggle": Interact with objects

## Movement Rules
**CRITICAL**: All movements are in ABSOLUTE directions (North/South/East/West).
- "up" = move North (upward on the image)
- "down" = move South (downward on the image)
- "left" = move West (leftward on the image)
- "right" = move East (rightward on the image)
- The robot will automatically rotate to face the correct direction before moving
- You don't need to think about the robot's current heading - just specify the direction you want to go

## Response Format
Respond in JSON format:
```json
{
    "action": "<action_name_or_number>",
    "environment_info": "<description of current state with spatial relationships in absolute coordinates (North/South/East/West)>",
    "reasoning": "<explanation of why you chose this action>"
}
```

**Important**: 
- Valid JSON format required
- Actions must be from the list above
- Complete mission from user prompt
- Use absolute directions (up/down/left/right), not relative to robot heading
- Think in terms of the image: up=North, down=South, left=West, right=East
"""
    
    def _get_default_user_prompt_template(self) -> str:
        """Return default user prompt template"""
        return "Based on the current image, choose the next action to complete the mission: {mission}. Use absolute directions (up/down/left/right)."
    
    def set_system_prompt(self, prompt: str):
        """Set system prompt"""
        self.system_prompt = prompt
    
    def set_user_prompt_template(self, template: str):
        """Set user prompt template"""
        self.user_prompt_template = template
    
    def get_user_prompt(self, mission: Optional[str] = None, **kwargs) -> str:
        """
        Build user prompt
        
        Args:
            mission: Mission text (use environment's mission if None)
            **kwargs: Keyword arguments to add to template
        
        Returns:
            Generated user prompt
        """
        if mission is None:
            state = self.env.get_state()
            mission = state.get('mission', 'explore')
        
        return self.user_prompt_template.format(mission=mission, **kwargs)
    
    def generate_action(
        self,
        user_prompt: Optional[str] = None,
        mission: Optional[str] = None
    ) -> Dict:
        """Generate an action using the VLM based on the current environment state.
        
        Captures the current environment image, constructs prompts, sends them
        to the VLM, and returns the parsed response. This is the main method for
        VLM-based action generation in robot control scenarios.
        
        Args:
            user_prompt: Custom user prompt. If None, the prompt is generated
                from the user_prompt_template using the mission text. Defaults to None.
            mission: Mission/objective text. Only used when user_prompt is None.
                If None, the mission is extracted from the environment state.
                Defaults to None.
        
        Returns:
            Dictionary containing the parsed VLM response. Typically includes:
                - 'action': str or list, the action(s) to execute
                - 'environment_info': str, description of the current state
                - 'reasoning': str, explanation of why this action was chosen
                - Additional fields as specified in required_fields
        
        Raises:
            RuntimeError: If the VLM API call fails.
            ValueError: If the VLM response cannot be parsed or is missing
                required fields.
        
        Examples:
            >>> from utils import MiniGridEmojiWrapper
            >>> from utils import VLMController
            >>> 
            >>> # Create environment and controller
            >>> env = MiniGridEmojiWrapper(size=10)
            >>> env.reset()
            >>> controller = VLMController(env=env)
            >>> 
            >>> # Generate action with default mission
            >>> response = controller.generate_action()
            >>> action = response['action']
            >>> print(f"VLM suggests: {action}")
            >>> 
            >>> # Generate action with custom mission
            >>> response = controller.generate_action(
            ...     mission="Go to the blue pillar and turn right"
            ... )
            >>> 
            >>> # Generate action with custom prompt
            >>> response = controller.generate_action(
            ...     user_prompt="Based on the image, what should I do next?"
            ... )
        
        Note:
            The VLM response is automatically parsed and validated. If parsing
            fails, a ValueError is raised with details about the failure.
        """
        image = self.env.get_image()
        
        if user_prompt is None:
            user_prompt = self.get_user_prompt(mission=mission)
        
        try:
            vlm_response_raw = self.vlm.generate(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            raise RuntimeError(f"VLM API call failed: {e}")
        
        try:
            vlm_response = self.postprocessor.process(vlm_response_raw, strict=True)
            return vlm_response
        except ValueError as e:
            raise ValueError(f"VLM response parsing failed: {e}\nOriginal response: {vlm_response_raw[:200]}...")
    
    def execute_action(self, action: Union[int, str]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action
        
        Args:
            action: Action (integer index or action name string)
        
        Returns:
            observation, reward, terminated, truncated, info
        
        Note:
            If environment has use_absolute_movement=True, step() automatically handles absolute movement.
            If environment provides parse_absolute_action method, it parses string actions.
        """
        # Use if environment provides parse_absolute_action
        if isinstance(action, str) and hasattr(self.env, 'parse_absolute_action'):
            try:
                action = self.env.parse_absolute_action(action)
            except (AttributeError, ValueError):
                pass  # Use original action if parsing fails
        
        return self.env.step(action)
    
    def step(
        self,
        user_prompt: Optional[str] = None,
        mission: Optional[str] = None
    ) -> Tuple[Dict, float, bool, bool, Dict, Dict]:
        """Generate an action using VLM and execute it in one step.
        
        This is a convenience method that combines generate_action() and
        execute_action() into a single call. It generates an action from the
        VLM based on the current environment state, then executes that action
        and returns both the environment step results and the VLM response.
        
        Args:
            user_prompt: Custom user prompt for the VLM. If None, the prompt
                is generated from the user_prompt_template using the mission.
                Defaults to None.
            mission: Mission/objective text. Only used when user_prompt is None.
                If None, extracted from the environment state. Defaults to None.
        
        Returns:
            Tuple containing:
                - observation: Dictionary or numpy array of the new environment state.
                - reward: float reward value from the executed action.
                - terminated: bool indicating if a terminal state was reached.
                - truncated: bool indicating if the episode was truncated.
                - info: Dictionary with additional step information.
                - vlm_response: Dictionary containing the parsed VLM response,
                    including 'action', 'reasoning', 'environment_info', etc.
        
        Raises:
            RuntimeError: If the VLM API call fails.
            ValueError: If the VLM response cannot be parsed.
            Exception: If action execution fails (depends on environment).
        
        Examples:
            >>> from utils import MiniGridEmojiWrapper, VLMController
            >>> 
            >>> # Create environment and controller
            >>> env = MiniGridEmojiWrapper(size=10)
            >>> env.reset()
            >>> controller = VLMController(env=env)
            >>> 
            >>> # Generate and execute action in one step
            >>> obs, reward, done, truncated, info, vlm_response = controller.step()
            >>> 
            >>> print(f"Action taken: {vlm_response['action']}")
            >>> print(f"Reasoning: {vlm_response.get('reasoning', 'N/A')}")
            >>> print(f"Reward: {reward}, Done: {done}")
            >>> 
            >>> # With custom mission
            >>> obs, reward, done, truncated, info, vlm_response = controller.step(
            ...     mission="Reach the goal"
            ... )
        
        Note:
            This method is ideal for simple control loops where you want to
            let the VLM decide and execute actions automatically. For more
            control, use generate_action() and execute_action() separately.
        """
        vlm_response = self.generate_action(user_prompt=user_prompt, mission=mission)
        action = vlm_response.get('action', 'up')
        
        obs, reward, terminated, truncated, info = self.execute_action(action)
        
        return obs, reward, terminated, truncated, info, vlm_response