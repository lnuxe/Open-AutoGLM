"""Main PhoneAgent class for orchestrating phone automation."""

import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.device_factory import get_device_factory
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._action_history: list[dict[str, Any]] = []  # For loop detection
        self._loop_break_count = 0  # How many times we've intervened to break a loop

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self._context = []
        self._step_count = 0

        # First step with user prompt
        result = self._execute_step(task, is_first=True)

        if result.finished:
            return result.message or "Task completed"

        # Continue until finished or max steps reached
        while self._step_count < self.agent_config.max_steps:
            result = self._execute_step(is_first=False)

            if result.finished:
                return result.message or "Task completed"

        return "Max steps reached"

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        return self._execute_step(task, is_first)

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._context = []
        self._step_count = 0
        self._action_history = []
        self._loop_break_count = 0

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        self._step_count += 1

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        current_app = device_factory.get_current_app(self.agent_config.device_id)

        # Build messages
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content, image_base64=screenshot.base64_data
                )
            )

        # Get model response
        try:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "=" * 50)
            print(f"\U0001f4ad {msgs['thinking']}:")
            print("-" * 50)
            response = self.model_client.request(self._context)
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            return StepResult(
                success=False,
                finished=True,
                action=None,
                thinking="",
                message=f"Model error: {e}",
            )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            # Parse failed — don't immediately finish! Instead, log the error
            # and ask the model to retry by injecting a hint into the context.
            if self.agent_config.verbose:
                traceback.print_exc()
            # Add a hint to the context so the model can retry
            self._context.append(
                MessageBuilder.create_assistant_message(
                    f"思考{response.thinking} 回复<answer>{response.action}</answer>"
                )
            )
            # Remove image from context to save space
            self._context[-2] = MessageBuilder.remove_images_from_message(self._context[-2])
            error_hint = (
                "你上一步的输出格式有误，无法解析为有效的操作指令。\n"
                "请严格按照以下格式输出，确保括号和引号完整匹配：\n"
                "  do(action=\"操作名\", 参数名=\"参数值\")\n"
                "或 finish(message=\"完成说明\")\n"
                "注意：不要在 message 参数中使用未转义的双引号，如需要引号请使用「」代替。"
            )
            return StepResult(
                success=False,
                finished=False,
                action=None,
                thinking="",
                message=error_hint,
            )

        if self.agent_config.verbose:
            # Print thinking process
            print("-" * 50)
            print(f"\U0001f3af {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"思考{response.thinking} 回复<answer>{response.action}</answer>"
            )
        )

        # Track action for loop detection (skip Note/Call_API/Wait since they don't change UI state)
        if action.get("_metadata") == "do":
            act_name = action.get("action")
            if act_name not in ("Note", "Call_API", "Wait",):
                sig = json.dumps(action, ensure_ascii=False, sort_keys=True)
                self._action_history.append({"action": action, "signature": sig})

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish

        # --- Loop detection: if stuck in a loop, inject a hint into the next step ---
        loop_detected = False
        loop_hint = None
        if not finished and len(self._action_history) >= 4:
            recent = self._action_history[-6:]
            sigs = [h["signature"] for h in recent]

            # Detect repeated action (same action 3+ consecutive times)
            for i in range(len(sigs) - 2):
                if sigs[i] == sigs[i + 1] == sigs[i + 2]:
                    loop_detected = True
                    loop_hint = (
                        "检测到你在重复执行相同的操作，这可能导致陷入循环。\n"
                        "请尝试以下方法之一：\n"
                        "1. 如果当前操作没有生效，请使用 Back 返回上一页，\n"
                        "   然后尝试不同的路径。\n"
                        "2. 如果页面内容没有变化，请尝试滑动页面看看是否有\n"
                        "   更多内容。\n"
                        "3. 如果确实找不到目标，请执行 finish(message=\"原因\")"
                    )
                    break

            # Detect A-B-A-B ping-pong pattern
            if not loop_detected and len(sigs) >= 4:
                for i in range(len(sigs) - 3):
                    if sigs[i] == sigs[i + 2] and sigs[i + 1] == sigs[i + 3] and sigs[i] != sigs[i + 1]:
                        loop_detected = True
                        loop_hint = (
                            "检测到你在两个页面之间来回切换，这可能导致陷入循环。\n"
                            "请尝试使用 Back 返回到之前的页面，然后尝试不同的路径。\n"
                            "如果目标页面无法到达，请执行 finish(message=\"原因\")。"
                        )
                        break

        if loop_detected:
            self._loop_break_count += 1
            if self.agent_config.verbose:
                print(f"\n\U0001f504 检测到循环行为 (第{self._loop_break_count}次干预)，正在注入提示...")
            # Replace the last assistant message to mark it as a failed attempt
            self._context.pop()
            self._context.append(
                MessageBuilder.create_assistant_message(
                    f"思考{response.thinking} 回复<answer>{response.action}</answer>"
                )
            )
            # Inject user hint for the next step
            self._context.append(
                MessageBuilder.create_user_message(
                    text=f"【系统提示】\n{loop_hint}",
                    image_base64=None,
                )
            )
            return StepResult(
                success=False,
                finished=False,
                action=action,
                thinking=response.thinking,
                message="Loop detected, injected hint for next step",
            )

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "\U0001f389 " + "=" * 48)
            print(
                f"\u2705 {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count