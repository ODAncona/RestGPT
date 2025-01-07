import time
import re
import logging
from typing import Type
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.callbacks import BaseCallbackManager
from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseChatModel

from langchain_community.utilities import RequestsWrapper

from .planner import Planner
from .api_selector import APISelector
from .caller import Caller
from .parser import ResponseParser
from utils import ReducedOpenAPISpec


logger = logging.getLogger(__name__)


class RestGPT(Chain):
    """Consists of an agent able to interact with APIs."""

    llm: BaseChatModel
    api_spec: ReducedOpenAPISpec
    planner: Planner
    api_selector: APISelector
    scenario: str = "tmdb"
    requests_wrapper: RequestsWrapper
    parser_class: Type[Chain] = ResponseParser
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"

    def __init__(
        self,
        llm: BaseChatModel,
        api_spec: ReducedOpenAPISpec,
        scenario: str,
        requests_wrapper: RequestsWrapper,
        parser_class: Type[Chain] = ResponseParser,
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        if scenario in ["TMDB", "Tmdb"]:
            scenario = "tmdb"
        if scenario in ["Spotify"]:
            scenario = "spotify"
        if scenario not in ["tmdb", "spotify"]:
            raise ValueError(f"Invalid scenario {scenario}")

        planner = Planner(llm=llm, scenario=scenario)
        api_selector = APISelector(
            llm=llm, scenario=scenario, api_spec=api_spec
        )

        init_args = {
            "llm": llm,
            "api_spec": api_spec,
            "planner": planner,
            "api_selector": api_selector,
            "scenario": scenario,
            "requests_wrapper": requests_wrapper,
            "parser_class": parser_class,
            "callback_manager": callback_manager,
            **kwargs,
        }
        super().__init__(**init_args)

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for RestGPT. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    @property
    def _chain_type(self) -> str:
        return "RestGPT"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return self.planner.output_keys

    def debug_input(self) -> str:
        print("Debug...")
        return input()

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        has_remaining_iterations = (
            self.max_iterations is None or iterations < self.max_iterations
        )
        has_remaining_time = (
            self.max_execution_time is None
            or time_elapsed < self.max_execution_time
        )

        return has_remaining_iterations and has_remaining_time

    def _return(self, output, intermediate_steps: list) -> Dict[str, Any]:
        self.callback_manager.on_agent_finish(
            output, color="green", verbose=self.verbose
        )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _get_api_selector_background(
        self, planner_history: List[Tuple[str, str]]
    ) -> str:
        if len(planner_history) == 0:
            return "No background"
        return "\n".join([step[1] for step in planner_history])

    def _should_continue_plan(self, plan) -> bool:
        return bool(re.search("Continue", plan))

    def _should_end(self, plan) -> bool:
        return bool(re.search("Final Answer", plan))

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        query = inputs["query"]

        planner_history: List[Tuple[str, str]] = []
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        # ~~~ Planner ~~~
        plan_result = self.planner.invoke(
            {"input": query, "history": planner_history}
        )
        plan = plan_result["result"]
        logger.info(f"Planner: {plan}")

        tmp_planner_history = [plan]
        api_selector_history: List[Tuple[str, str, str]] = []
        api_selector_background = self._get_api_selector_background(
            planner_history
        )
        while self._should_continue(iterations, time_elapsed):

            # ~~~ API Selector ~~~
            api_plan = self.api_selector.invoke(
                {
                    "plan": plan,
                    "background": api_selector_background,
                }
            )["result"]
            finished = re.match(r"No API call needed.(.*)", api_plan)

            # ~~~ Caller ~~~
            if not finished:
                executor = Caller(
                    llm=self.llm,
                    api_spec=self.api_spec,
                    scenario=self.scenario,
                    parser_class=self.parser_class,
                    requests_wrapper=self.requests_wrapper,
                )
                execution_res = executor.invoke(
                    {
                        "api_plan": api_plan,
                        "background": api_selector_background,
                    }
                )["result"]
            else:
                execution_res = finished.group(1)

            # ~~~ Log Execution Result ~~~
            planner_history.append((plan, execution_res))
            api_selector_history.append((plan, api_plan, execution_res))

            # ~~~ Planner ~~~
            plan = self.planner.invoke(
                {"input": query, "history": planner_history}
            )["result"]
            logger.info(f"Planner: {plan}")

            # while self._should_continue_plan(plan):
            #     api_selector_background = self._get_api_selector_background(
            #         planner_history
            #     )
            #     api_plan = self.api_selector.invoke(
            #         {
            #             "plan": tmp_planner_history[0],
            #             "background": api_selector_background,
            #             "history": api_selector_history,
            #             "instruction": plan,
            #         }
            #     )["result"]
            #     finished = re.match(r"No API call needed.(.*)", api_plan)
            #     if not finished:
            #         executor = Caller(
            #             llm=self.llm,
            #             api_spec=self.api_spec,
            #             scenario=self.scenario,
            #             parser_class=self.parser_class,
            #             requests_wrapper=self.requests_wrapper,
            #         )
            #         execution_res = executor.invoke(
            #             {
            #                 "api_plan": api_plan,
            #                 "background": api_selector_background,
            #             }
            #         )["result"]
            #     else:
            #         execution_res = finished.group(1)

            #     planner_history.append((plan, execution_res))
            #     api_selector_history.append((plan, api_plan, execution_res))

            #     plan = self.planner.invoke(
            #         {
            #             "input": query,
            #             "history": planner_history,
            #         }
            #     )["result"]
            #     logger.info({f"Planner HOHO: {planner_history}"})
            #     logger.info(f"Planner HOHO: {plan}")

            if self._should_end(plan):
                break

            iterations += 1
            time_elapsed = time.time() - start_time

        return {"result": plan}
