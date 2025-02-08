"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-09-27
version: 1.4
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

import logging
import os
import uuid
from typing import List, Optional

from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError
from pydantic import BaseModel

from utils.pipelines.main import get_last_assistant_message

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str
        log_level: str = "INFO"

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            }
        )
        self.langfuse = None
        self.chat_traces = {}
        self.chat_generations = {}
        logger.setLevel(self.valves.log_level)

    async def on_startup(self):
        logger.info("Starting up Langfuse Filter Pipeline")
        self.set_langfuse()

    async def on_shutdown(self):
        logger.info("Shutting down Langfuse Filter Pipeline")
        self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            self.langfuse.auth_check()
        except UnauthorizedError:
            logger.error(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            logger.error(
                f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings."
            )

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        logger.info(f"Processing inlet pipeline: {__name__}")
        logger.debug(f"Processing request body: {body}")
        logger.debug(f"User context: {user}")

        # Skip processing if the request is a task
        is_task: bool = body.get("metadata", {}).get("task", "") != ""

        if is_task:
            return body

        # Check for presence of required keys and generate chat_id if it is needed
        chat_id = body.get("chat_id") or body.get("metadata", {}).get("chat_id")

        if not chat_id:
            chat_id = str(uuid.uuid4())  # Generate a unique UUID for chat_id
            body["metadata"]["chat_id"] = chat_id  # Add it to metadata

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]

        if missing_keys:
            error_message = (
                f"Missing keys in the request body: {', '.join(missing_keys)}"
            )
            logger.error(error_message)
            raise ValueError(error_message)

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["email"],
            metadata={"user_name": user["name"], "user_id": user["id"]},
            session_id=chat_id,
        )

        generation = trace.generation(
            name=chat_id,
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui"},
        )

        self.chat_traces[chat_id] = trace
        self.chat_generations[chat_id] = generation

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        logger.info(f"Processing outlet pipeline: {__name__}")
        logger.debug(f"Processing response body: {body}")

        # Skip processing if the request is a task
        is_task: bool = body.get("metadata", {}).get("task", "") != ""

        if is_task:
            return body

        # Define chat_id as a variable
        chat_id = body.get("chat_id") or body.get("metadata", {}).get("chat_id")

        if chat_id not in self.chat_generations or chat_id not in self.chat_traces:
            logger.warning("Langfuse trace not found for this chat_id in outlet")
            return body

        trace = self.chat_traces[chat_id]
        generation = self.chat_generations[chat_id]
        assistant_message = get_last_assistant_message(body["messages"])

        # Extract usage information for models that support it
        usage = None
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        if assistant_message_obj:
            info = assistant_message_obj.get("info", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get(
                    "prompt_tokens"
                )
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }

        # Update generation
        trace.update(
            output=assistant_message,
        )
        generation.end(
            output=assistant_message,
            metadata={"interface": "open-webui"},
            usage=usage,
        )

        # Clean up the chat_generations dictionary
        if chat_id in self.chat_traces:
            del self.chat_traces[chat_id]

        if chat_id in self.chat_generations:
            del self.chat_generations[chat_id]

        return body
