"""
Chat API endpoints.
"""
import logging
import uuid
from typing import List
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.ai_service import ai_service
from app.memory.conversation_memory import conversation_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for user interactions.
    
    Handles:
    - Creating new conversations or continuing existing ones
    - Maintaining conversation history via ConversationMemory
    - Generating AI responses
    
    Args:
        request: Chat request containing message and optional conversation_id
        
    Returns:
        ChatResponse with AI response and conversation_id
    """
    try:
        # Get or create session_id (using conversation_id from request)
        session_id = request.conversation_id or str(uuid.uuid4())
        
        # Retrieve conversation history from memory
        history = conversation_memory.get_history(session_id)
        
        # Create and add user message
        user_message = ChatMessage(role="user", content=request.message)
        conversation_memory.add_message(session_id, user_message)
        
        # Build message list for AI service (include history + new user message)
        messages_for_ai = history + [user_message]
        
        # Generate AI response
        response_text = await ai_service.generate_response(
            messages=messages_for_ai,
            user_id=request.user_id
        )
        
        # Create and store assistant response
        assistant_message = ChatMessage(role="assistant", content=response_text)
        conversation_memory.add_message(session_id, assistant_message)
        
        logger.info(
            f"Chat request processed: session_id={session_id}, "
            f"user_id={request.user_id}, "
            f"message_count={conversation_memory.get_message_count(session_id)}"
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id=session_id,
            metadata={
                "provider": ai_service.provider,
                "message_count": conversation_memory.get_message_count(session_id)
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    """
    Retrieve conversation history by ID.
    
    Args:
        conversation_id: Unique conversation identifier (session_id)
        
    Returns:
        Dictionary containing conversation messages
    """
    if not conversation_memory.has_session(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    history = conversation_memory.get_history(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ],
        "message_count": len(history)
    }
