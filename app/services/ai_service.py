"""
AI/LLM service layer with provider abstraction.
Supports multiple LLM providers (mock, OpenAI, etc.)
"""
import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import ChatMessage
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
        
        Args:
            messages: List of chat messages (conversation history)
            user_id: Optional user identifier for personalization
            
        Returns:
            Generated response string
        """
        if self.provider == "mock":
            return self._mock_response(messages)
        elif self.provider == "openai":
            return await self._openai_response(messages)
        else:
            logger.warning(f"Unknown provider {self.provider}, using mock")
            return self._mock_response(messages)
    
    def _mock_response(self, messages: List[ChatMessage]) -> str:
        """
        Mock AI response for development/testing.
        Provides basic conversational responses without calling an LLM.
        """
        # Get the last user message
        user_messages = [msg for msg in messages if msg.role == "user"]
        if not user_messages:
            return "Hello! How can I help you today?"
        
        last_message = user_messages[-1].content.lower()
        
        # Simple pattern matching for demo
        if any(word in last_message for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your personal AI assistant. How can I help you today?"
        elif any(word in last_message for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! Feel free to reach out anytime you need assistance."
        elif "?" in last_message:
            return "That's an interesting question! I'm currently running in mock mode. Once connected to a real LLM, I'll be able to provide detailed answers to your questions."
        elif any(word in last_message for word in ["help", "what can you do"]):
            return "I can help you with various tasks like answering questions, taking notes, setting reminders, and more. What would you like to do?"
        else:
            return f"I understand you said: '{user_messages[-1].content}'. I'm currently in mock mode, but I'm ready to assist you with more advanced capabilities once connected to a real LLM!"
    
    async def _openai_response(self, messages: List[ChatMessage]) -> str:
        """
        Generate response using OpenAI API.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Generated response string
        """
        try:
            from openai import AsyncOpenAI
            
            if not config.settings.openai_api_key:
                logger.error("OpenAI API key not configured")
                return "Error: OpenAI API key is not configured. Please set OPENAI_API_KEY environment variable."
            
            client = AsyncOpenAI(api_key=config.settings.openai_api_key)
            
            # Convert ChatMessage to OpenAI format
            formatted_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Add system message if not present
            if not any(msg.role == "system" for msg in messages):
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": self.system_personality
                })
            
            response = await client.chat.completions.create(
                model=config.settings.openai_model,
                messages=formatted_messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content or "I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"Sorry, I encountered an error while generating a response: {str(e)}"


# Global service instance
ai_service = AIService()
