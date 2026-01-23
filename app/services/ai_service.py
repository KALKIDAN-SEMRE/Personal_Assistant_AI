"""
AI/LLM service layer with provider abstraction.
Supports multiple LLM providers (mock, OpenAI, Ollama).
"""

import logging
from typing import List, Optional
import requests

from app.models.schemas import ChatMessage
from app.memory.semantic import semantic_memory
import config

logger = logging.getLogger(__name__)


class AIService:
    """Abstracted AI service for generating responses."""

    def __init__(self):
        self.provider = config.settings.llm_provider
        self.system_personality = config.settings.system_personality

    async def generate_response(
        self,
        messages: List[ChatMessage],
        user_id: Optional[str] = None
    ) -> str:
        """
        Generate AI response from conversation messages.
        Includes semantic memory retrieval for personalized responses.

        Args:
            messages: List of chat messages (conversation history)
            user_id: Optional user identifier for personalization

        Returns:
            Generated response string
        """
        # Get user's latest message for semantic memory retrieval
        user_messages = [msg for msg in messages if msg.role == "user"]
        query_text = user_messages[-1].content if user_messages else ""
        
        # Retrieve relevant semantic memories
        semantic_context = ""
        if user_id and query_text:
            try:
                semantic_context = await semantic_memory.get_context_for_query(
                    query=query_text,
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Error retrieving semantic memories: {e}")
        
        # Enhance system personality with semantic context
        enhanced_personality = self.system_personality
        if semantic_context:
            enhanced_personality = self.system_personality + semantic_context
        
        if self.provider == "mock":
            return self._mock_response(messages, enhanced_personality)

        elif self.provider == "openai":
            return await self._openai_response(messages, enhanced_personality)

        elif self.provider == "ollama":
            return await self._ollama_response(messages, enhanced_personality)

        else:
            logger.warning(f"Unknown provider '{self.provider}', falling back to mock.")
            return self._mock_response(messages, enhanced_personality)

    # ------------------------------------------------------------------
    # MOCK PROVIDER
    # ------------------------------------------------------------------

    def _mock_response(self, messages: List[ChatMessage], enhanced_personality: str = None) -> str:
        """Mock AI response for development/testing."""
        personality = enhanced_personality or self.system_personality

        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return "Hello! How can I help you today?"

        last_message = user_messages[-1].content.lower()

        if any(word in last_message for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your personal AI assistant. How can I help you today?"

        elif any(word in last_message for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! Feel free to reach out anytime you need assistance."

        elif "?" in last_message:
            return (
                "That's a great question! I'm currently running in mock mode. "
                "Once connected to a real LLM, I'll give you detailed answers."
            )

        elif any(word in last_message for word in ["help", "what can you do"]):
            return (
                "I can help you answer questions, take notes, set reminders, "
                "and much more. What would you like to do?"
            )

        return (
            f"I understand you said: '{user_messages[-1].content}'. "
            "I'm currently in mock mode, but I'm ready to do more once connected "
            "to a real AI model."
        )
    
    async def _openai_response(self, messages: List[ChatMessage], enhanced_personality: str = None) -> str:
        """Generate response using OpenAI API."""
        try:
            from openai import AsyncOpenAI

            if not config.settings.openai_api_key:
                logger.error("OpenAI API key not configured")
                return "Error: OpenAI API key is not configured."

            client = AsyncOpenAI(api_key=config.settings.openai_api_key)
            personality = enhanced_personality or self.system_personality

            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]

            if not any(msg.role == "system" for msg in messages):
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": personality
                })

            response = await client.chat.completions.create(
                model=config.settings.openai_model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content or "No response generated."

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I encountered an error while generating a response."
    
    async def _ollama_response(self, messages: List[ChatMessage], enhanced_personality: str = None) -> str:
        """Generate response using local Ollama LLM."""
        try:
            personality = enhanced_personality or self.system_personality
            prompt = self._format_messages_for_ollama(messages, personality)

            payload = {
                "model": config.settings.ollama_model,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                config.settings.ollama_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            return response.json().get("response", "No response generated.")

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return "Sorry, I had trouble communicating with the local AI model."

    def _format_messages_for_ollama(self, messages: List[ChatMessage], personality: str = None) -> str:
        """
        Convert chat messages into a single prompt suitable for local LLMs.
        """
        system_prompt = personality or self.system_personality
        prompt = f"System: {system_prompt}\n\n"

        for msg in messages:
            role = msg.role.capitalize()
            prompt += f"{role}: {msg.content}\n"

        prompt += "Assistant:"
        return prompt


# Global service instance
ai_service = AIService()
