import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from copy import deepcopy
import yaml
import time
import re
import requests
from pydantic import BaseModel, Field
import tiktoken

from langchain.chains.base import Chain
from langchain_community.utilities import TextRequestsWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel

from utils import (
    get_matched_endpoint,
    ReducedOpenAPISpec,
)
from .parser import ResponseParser


logger = logging.getLogger(__name__)


class Caller_Message(BaseModel):
    """Caller message schema."""

    api_plan: str = Field(..., title="API plan")
    background: Optional[str] = Field(None, title="Background information")
    agent_scratchpad: Optional[str] = Field(None, title="Agent scratchpad")


icl_examples = [
    """Operation: POST
Input: {{
"url": "https://api.twitter.com/2/tweets",
"params": {{
    "tweet.fields": "created_at"
}}
"data": {{
    "text": "Hello world!"
}},
"description": "The API response is a twitter object.",
"output_instructions": "What is the id of the new twitter?"
}}""",
    """
Operation: GET
Input: {{
    "url": "https://api.themoviedb.org/3/person/5026/movie_credits",
    "description": "The API response is the movie credit list of Akira Kurosawa (id 5026)",
    "output_instructions": "What are the names and ids of the movies directed by this person?"
}}""",
    """
Operation: PUT
Input: {{
    "url": "https://api.spotify.com/v1/me/player/volume",
    "params": {{
        "volume_percent": "20"
    }},
    "description": "Set the volume for the current playback device."
}}""",
]

CALLER_PROMPT = """You are an agent that gets an API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.
Your task is to complete the corresponding api calls according to the plan.


Here is documentation on the API:
Base url: {api_url}
Endpoints: {api_docs}

If the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url.

You can use http request method, i.e., GET, POST, DELETE, PATCH, PUT, and generate the corresponding parameters according to the API documentation and the plan.
The input should be a JSON string which has 3 base keys: url, description, output_instructions
The value of "url" should be a string.
The value of "description" should describe what the API response is about. The description should be specific.
The value of "output_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the POST request creates. Note "output_instructions" MUST be natural language and as verbose as possible! It cannot be "return the full response". Output instructions should faithfully contain the contents of the api calling plan and be as specific as possible. The output instructions can also contain conditions such as filtering, sorting, etc.
If you are using GET method, add "params" key, and the value of "params" should be a dict of key-value pairs.
If you are using POST, PATCH or PUT methods, add "data" key, and the value of "data" should be a dict of key-value pairs.
Remember to add a comma after every value except the last one, ensuring that the overall structure of the JSON remains valid.

{icl_examples}

I will give you the background information and the plan you should execute.
Background: background information which you can use to execute the plan, e.g., the id of a person.
Plan: the plan of API calls to execute

If you have completed all steps in the plan and have nothing else to do, produce:
Execution Result: <final summary of the entire plan's execution>

Follow this format strictly and do not add any additional information.

Plan: {api_plan}
Background: {background}
Thought: you should always think about what to do
Operation: the request method to take, should be one of the following: GET, POST, DELETE, PATCH, PUT
Input: the input to the operation in json format

Begin!
"""


class Caller(Chain):
    llm: BaseChatModel
    api_spec: ReducedOpenAPISpec
    scenario: str
    requests_wrapper: TextRequestsWrapper
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    parser_class: Type[Chain] = ResponseParser
    with_response: bool = False
    output_key: str = "result"
    endpoint_docs_by_name: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        llm: BaseChatModel,
        api_spec: ReducedOpenAPISpec,
        scenario: str,
        requests_wrapper: TextRequestsWrapper,
        parser_class: Type[Chain] = ResponseParser,
        with_response: bool = False,
        **kwargs: Any,
    ) -> None:
        init_args = {
            "llm": llm,
            "api_spec": api_spec,
            "scenario": scenario,
            "requests_wrapper": requests_wrapper,
            "parser_class": parser_class,
            "with_response": with_response,
            **kwargs,
        }
        super().__init__(**init_args)
        self.endpoint_docs_by_name = {
            name: docs for name, _, docs in self.api_spec.endpoints
        }

    @property
    def _chain_type(self) -> str:
        return "RestGPT Caller"

    @property
    def input_keys(self) -> List[str]:
        return ["api_plan"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        has_remaining_iterations = (
            self.max_iterations is None or iterations < self.max_iterations
        )
        has_remaining_time = (
            self.max_execution_time is None
            or time_elapsed < self.max_execution_time
        )

        return has_remaining_iterations and has_remaining_time

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought: "

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(self, history: List[Tuple[str, str]]) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += f"Thought {i}: {plan}"
            scratchpad += f"{self.observation_prefix}{execution_res}\n"
        return scratchpad

    def _get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        if "Execution Result:" in llm_output:
            return (
                "DONE",
                llm_output.split("Execution Result:")[-1].strip(),
            )
        # \s matches against tab/newline/whitespace
        regex = r"Operation:[\s]*(.*?)[\n]*Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # TODO: not match, just return
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        if action not in ["GET", "POST", "DELETE", "PUT"]:
            raise NotImplementedError

        action_input = action_input.strip().strip("`")
        left_bracket = action_input.find("{")
        right_bracket = action_input.rfind("}")
        action_input = action_input[left_bracket : right_bracket + 1]
        # action_input = fix_json_error(action_input)

        # avoid error in the JSON format
        return action, action_input

    def _get_response(self, action: str, action_input: str) -> str:
        try:
            data = json.loads(action_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse action input: {action_input}")

        desc = data.get("description", "No description")
        query = data.get("output_instructions", None)

        params, request_body = None, None
        if action == "GET":
            if "params" in data:
                params = data.get("params")
                response = self.requests_wrapper.get(
                    data.get("url"), params=params
                )
            else:
                response = self.requests_wrapper.get(data.get("url"))
        elif action == "POST":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.post(
                data["url"], params=params, data=request_body
            )
        elif action == "PUT":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.put(
                data["url"], params=params, data=request_body
            )
        elif action == "DELETE":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.delete(
                data["url"], params=params, json=request_body
            )
        else:
            raise NotImplementedError(f"Unsupported action: {action}")

        if isinstance(response, requests.models.Response):
            if response.status_code != 200:
                return response.text
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            raise NotImplementedError

        return response_text, params, request_body, desc, query

    def _prepare_api_docs(self, api_plan) -> Tuple[str, str]:
        api_url = self.api_spec.servers[0]["url"]
        matched_endpoints = get_matched_endpoint(self.api_spec, api_plan)
        if matched_endpoints is None:
            return "", ""
            raise ValueError(
                f"Could not find a matching endpoint for the API plan: {api_plan}"
            )

        # assert (
        #     len(matched_endpoints) == 1
        # ), f"Found {len(matched_endpoints)} matched endpoints, but expected 1."
        endpoint_name = matched_endpoints[-1]
        tmp_docs = deepcopy(self.endpoint_docs_by_name.get(endpoint_name))

        # Traitez les "responses" dans tmp_docs
        if "responses" in tmp_docs and "content" in tmp_docs["responses"]:
            content = tmp_docs["responses"]["content"]
            if "application/json" in content:
                tmp_docs["responses"] = content["application/json"]["schema"][
                    "properties"
                ]
            elif "application/json; charset=utf-8" in content:
                tmp_docs["responses"] = content[
                    "application/json; charset=utf-8"
                ]["schema"]["properties"]
        if not self.with_response and "responses" in tmp_docs:
            tmp_docs.pop("responses")

        # Limitez la taille de tmp_docs
        tmp_docs = yaml.dump(tmp_docs)
        encoder = tiktoken.encoding_for_model("gpt-4o")
        encoded_docs = encoder.encode(tmp_docs)
        if len(encoded_docs) > 1500:
            tmp_docs = encoder.decode(encoded_docs[:1500])

        api_doc_for_caller = f"== Docs for {endpoint_name} == \n{tmp_docs}\n"
        return api_doc_for_caller, api_url

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:

        api_plan = inputs["api_plan"]
        api_doc_for_caller, api_url = self._prepare_api_docs(api_plan)

        caller_prompt = PromptTemplate(
            template=CALLER_PROMPT,
            partial_variables={
                "api_url": api_url,
                "api_docs": api_doc_for_caller,
                "icl_examples": "\n".join(icl_examples),
            },
            input_variables=["api_plan", "background", "agent_scratchpad"],
        )

        # Initialize the first LLM chain
        caller_chain = caller_prompt | self.llm

        # **First LLM Call:** Generate Operation and Input
        caller_chain_output = caller_chain.invoke(
            {
                "api_plan": api_plan,
                "background": inputs.get("background", ""),
                "stop": self._stop,
            }
        ).content
        logger.info(f"Caller: {caller_chain_output}")

        # Parse the operation and input
        action, action_input = self._get_action_and_input(caller_chain_output)
        if action == "DONE":
            return {"result": action_input}

        # **Execute the API Call**
        response, params, request_body, desc, query = self._get_response(
            action, action_input
        )
        response = response[:7500]

        called_endpoint_name = (
            action + " " + json.loads(action_input)["url"].replace(api_url, "")
        )
        called_endpoint_name = get_matched_endpoint(
            self.api_spec, called_endpoint_name
        )[0]
        api_path = api_url + called_endpoint_name.split(" ")[-1]
        api_doc_for_parser = self.endpoint_docs_by_name.get(
            called_endpoint_name
        )

        matched_endpoints = get_matched_endpoint(self.api_spec, api_plan)
        endpoint_name = matched_endpoints[0]

        if self.scenario == "spotify" and endpoint_name == "GET /search":
            if params is not None and "type" in params:
                search_type = params["type"] + "s"
            else:
                params_in_url = json.loads(action_input)["url"].split("&")
                for param in params_in_url:
                    if "type=" in param:
                        search_type = param.split("=")[-1] + "s"
                        break
            api_doc_for_parser["responses"]["content"]["application/json"][
                "schema"
            ]["properties"] = {
                search_type: api_doc_for_parser["responses"]["content"][
                    "application/json"
                ]["schema"]["properties"][search_type]
            }

        # **Parse the API Response**
        api_response_parser = self.parser_class(
            llm=self.llm,
            api_path=api_path,
            api_doc=api_doc_for_parser,
        )

        params_or_data = {
            "params": params if params else "No parameters",
            "data": request_body if request_body else "No request body",
        }

        parsing_res = api_response_parser.invoke(
            {
                "query": query,
                "response_description": desc,
                "api_param": params_or_data,
                "json": response,
            }
        )["result"]
        logger.info(f"Parser:\n{parsing_res}")

        execution_res = f"{caller_chain_output}\nObservation: {parsing_res}"

        return {"result": execution_res}
