import os
from pydantic import BaseModel, Field
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the local research agent."""

    query_generator_model: str = Field(
        default="groq:llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for generating search keywords."
        },
    )

    answer_model: str = Field(
        default="groq:llama-3.3-70b-versatile",
        metadata={
            "description": "The name of the language model to use for the final technical answer."
        },
    )

    number_of_initial_queries: int = Field(
        default=2, # Зменшено до 2 для економії токенів у лінійному графі
        metadata={"description": "The number of search queries (keyword sets) to generate."},
    )

    local_dir: str = Field(
        default="/Users/ostapzherebtsov/Desktop/langgraph_old_docs",
        metadata={"description": "The directory to search for local .md files."}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Отримуємо значення з системних змінних (наприклад, LOCAL_DIR) або з конфігурації LangGraph
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Відфільтровуємо None значення, щоб використати default з Field
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)