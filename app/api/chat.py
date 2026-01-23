"""
Chat API endpoints.
"""
import logging
import uuid
from typing import List
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.ai_service import ai_service
from app.memory.persistent import persistent_memory
from app.memory.semantic import semantic_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for user interactions.
    
    Handles:
    - Creating new conversations or continuing existing ones
    - Maintaining conversation history via PersistentConversationMemory
    - Generating AI responses
    - Persisting all messages to database
    
    Args:
        request: Chat request containing message and optional conversation_id
        
    Returns:
        ChatResponse with AI response and conversation_id
    """
    try:
        # Get or create session_id (using conversation_id from request)
        session_id = request.conversation_id or str(uuid.uuid4())
        
        # Load conversation history from persistent storage
        # This loads the last N messages (enforced by max_history)
        history = persistent_memory.get_recent_messages(session_id)
        
        # Create user message
        user_message = ChatMessage(role="user", content=request.message)
        
        # Save user message to persistent storage
        persistent_memory.save_message(session_id, "user", request.message)
        
        # Build message list for AI service (history + new user message)
        messages_for_ai = history + [user_message]
        
        # Generate AI response
        response_text = await ai_service.generate_response(
            messages=messages_for_ai,
            user_id=request.user_id
        )
        
        # Save assistant response to persistent storage
        persistent_memory.save_message(session_id, "assistant", response_text)
        
        # Extract and store semantic memories from the conversation
        if request.user_id:
            try:
                # Use the full conversation including the new messages
                full_conversation = messages_for_ai + [
                    ChatMessage(role="assistant", content=response_text)
                ]
                await semantic_memory.extract_and_store(
                    messages=full_conversation,
                    user_id=request.user_id
                )
            except Exception as e:
                logger.warning(f"Error extracting semantic memories: {e}")
        
        # Get total message count for metadata
        total_count = persistent_memory.get_message_count(session_id)
        
        logger.info(
            f"Chat request processed: session_id={session_id}, "
            f"user_id={request.user_id}, "
            f"message_count={total_count}"
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id=session_id,
            metadata={
                "provider": ai_service.provider,
                "message_count": total_count
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    """
    Retrieve conversation history by ID from persistent storage.
    
    Args:
        conversation_id: Unique conversation identifier (session_id)
        
    Returns:
        Dictionary containing conversation messages
    """
    if not persistent_memory.has_session(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Retrieve recent messages (respects max_history limit)
    history = persistent_memory.get_recent_messages(conversation_id)
    total_count = persistent_memory.get_message_count(conversation_id)
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ],
        "message_count": len(history),
        "total_messages": total_count
    }
