"""Gemini LLM service implementation."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, cast

import google.generativeai as genai
from pydantic import BaseModel, ValidationError

from fast_graphrag._llm._base import BaseLLMService
from fast_graphrag._models import BaseModelAlias

T_model = TypeVar("T_model", bound=BaseModel)

@dataclass
class GeminiLLMService(BaseLLMService):
    """Gemini implementation for LLM services."""
    model: Optional[str] = field(default="gemini-1.0-pro")
    api_key: Optional[str] = field(default=None)
    temperature: float = field(default=0.6)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    llm_calls_count: int = field(default=0, init=False)
    _client: Optional[Any] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize after dataclass initialization."""
        if self.model is None:
            raise ValueError("Model name must be provided.")
        
        # Initialize Gemini client
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided either in constructor or as GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(self.model)

    async def send_message(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict[str, str]]] = None,
        response_model: Optional[Type[T_model]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[T_model, List[dict[str, str]]]:
        """Send a message to the Gemini LLM and handle the response.

        Args:
            prompt: Main input text
            model: Optional model override
            system_prompt: Optional system instructions
            history_messages: Previous conversation messages
            response_model: Expected response structure
            temperature: Optional temperature override
            **kwargs: Additional parameters

        Returns:
            Tuple of (parsed response, message history)
        """
        temperature = temperature or self.temperature
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")

        messages: List[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history_messages:
            messages.extend(history_messages)

        messages.append({"role": "user", "content": prompt})

        # Add format instruction to the prompt if response_model exists
        if response_model:
            model_class = (response_model.Model 
                        if issubclass(response_model, BaseModelAlias)
                        else response_model)
            schema = cast(Dict[str, Any], model_class.model_json_schema())
            schema_instruction = (
                "IMPORTANT: Your response must be a valid JSON object following this schema:\n"
                f"{schema}\n\n"
                "Example of a valid response format:\n"
                "{\n"
                '    "entities": [\n'
                '        {"name": "Scrooge", "type": "person", "desc": "A miserly businessman"},\n'
                '        {"name": "London", "type": "location", "desc": "The city where the story takes place"}\n'
                '    ],\n'
                '    "relationships": [\n'
                '        {"source": "Scrooge", "target": "London", "desc": "Scrooge lives and works in London"}\n'
                '    ],\n'
                '    "other_relationships": []\n'
                "}"
            )
            messages.insert(0, {"role": "system", "content": schema_instruction})

        # Combine messages into a single prompt
        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # Generate response
        if not self._client:
            raise ValueError("Gemini client not initialized.")
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            candidate_count=1
        )
        
        response = self._client.generate_content(
            combined_prompt,
            generation_config=generation_config
        )

        # Extract text content from response
        response_text = response.text

        # Parse response using response_model if provided
        try:
            if response_model:
                if issubclass(response_model, BaseModelAlias):
                    llm_response = cast(T_model, response_model.Model.model_validate_json(response_text))
                else:
                    llm_response = cast(T_model, response_model.model_validate_json(response_text))
            else:
                llm_response = cast(T_model, response_text)
        except ValidationError as e:
            raise ValueError(f"Invalid JSON response: {str(e)}") from e

        self.llm_calls_count += 1

        if not llm_response:
            raise ValueError("No response received from the language model.")

        messages.append({
            "role": "assistant",
            "content": (llm_response.model_dump_json() 
                      if isinstance(llm_response, BaseModel) 
                      else str(llm_response)),
        })

        if response_model and issubclass(response_model, BaseModelAlias):
            llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

        return llm_response, messages