"""
Chat API endpoints.
"""
import logging
import uuid
from typing import List
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, ChatMessage
from app.services.ai_service import ai_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# In-memory conversation storage (will be replaced with database later)
conversations: dict[str, List[ChatMessage]] = {}


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for user interactions.
    
    Handles:
    - Creating new conversations or continuing existing ones
    - Maintaining conversation history
    - Generating AI responses
    
    Args:
        request: Chat request containing message and optional conversation_id
        
    Returns:
        ChatResponse with AI response and conversation_id
    """
    try:
        # Get or create conversation
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Initialize conversation if new
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message to conversation
        user_message = ChatMessage(role="user", content=request.message)
        conversations[conversation_id].append(user_message)
        
        # Generate AI response
        response_text = await ai_service.generate_response(
            messages=conversations[conversation_id],
            user_id=request.user_id
        )
        
        # Add assistant response to conversation
        assistant_message = ChatMessage(role="assistant", content=response_text)
        conversations[conversation_id].append(assistant_message)
        
        logger.info(
            f"Chat request processed: conversation_id={conversation_id}, "
            f"user_id={request.user_id}"
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            metadata={
                "provider": ai_service.provider,
                "message_count": len(conversations[conversation_id])
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
        conversation_id: Unique conversation identifier
        
    Returns:
        Dictionary containing conversation messages
    """
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "conversation_id": conversation_id,
        "messages": [
            {"role": msg.role, "content": msg.content}
            for msg in conversations[conversation_id]
        ],
        "message_count": len(conversations[conversation_id])
    }
