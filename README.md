# Personal AI Assistant

A production-ready, extensible personal AI assistant built with FastAPI. Designed for local development with cloud deployment capabilities.

## 🎯 Features

- **Natural Chat Interface**: RESTful API for conversational interactions
- **Conversation Memory**: Maintains context across multiple exchanges
- **Provider Abstraction**: Easy switching between mock, OpenAI, and future LLM providers
- **Clean Architecture**: Modular, scalable design with separation of concerns
- **Type-Safe**: Full type hints and Pydantic validation
- **Extensible**: Built-in framework for adding tools and capabilities

## 🏗️ Architecture

```
/assistant
  /app
    /api          # FastAPI routes and endpoints
    /core         # Core application components
    /services     # Business logic (AI service, etc.)
    /memory       # Conversation and memory management
    /models       # Data models and schemas
    /utils        # Utility functions
  /tests          # Test suite
  main.py         # Application entry point
  config.py       # Configuration management
  requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or poetry

### Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables** (optional)

Create a `.env` file in the root directory:

```env
DEBUG=True
HOST=127.0.0.1
PORT=8000
LLM_PROVIDER=mock
OPENAI_API_KEY=your-api-key-here  # If using OpenAI
OPENAI_MODEL=gpt-3.5-turbo
SYSTEM_PERSONALITY=You are a helpful, friendly, and intelligent personal assistant.
```

5. **Run the application**

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

## 📚 API Documentation

Once the server is running, visit:

- **Interactive API Docs (Swagger)**: http://127.0.0.1:8000/docs
- **Alternative Docs (ReDoc)**: http://127.0.0.1:8000/redoc

## 💬 Usage Examples

### Chat Endpoint

**POST** `/api/v1/chat`

Request body:
```json
{
  "message": "Hello, how can you help me?",
  "conversation_id": null,  // Optional: for continuing conversations
  "user_id": "default"       // Optional: for personalization
}
```

Response:
```json
{
  "response": "Hello! I'm your personal AI assistant...",
  "conversation_id": "uuid-here",
  "metadata": {
    "provider": "mock",
    "message_count": 2
  }
}
```

### Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What can you do?"}'
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/api/v1/chat",
    json={"message": "Hello!"}
)

data = response.json()
print(data["response"])
```

## 🔧 Configuration

Configuration is managed through environment variables or a `.env` file. Key settings:

- `LLM_PROVIDER`: AI provider (`mock` or `openai`)
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI provider)
- `OPENAI_MODEL`: Model name (default: `gpt-3.5-turbo`)
- `SYSTEM_PERSONALITY`: System prompt for the assistant
- `DEBUG`: Enable debug mode (default: `False`)
- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)

## 🧪 Testing

Run tests (once test suite is expanded):

```bash
pytest
```

## 📦 Project Structure

- **`main.py`**: FastAPI application entry point
- **`config.py`**: Configuration management with environment variables
- **`app/api/chat.py`**: Chat endpoint implementation
- **`app/services/ai_service.py`**: AI/LLM service layer with provider abstraction
- **`app/models/schemas.py`**: Pydantic models for request/response validation

## 🛣️ Roadmap

### Phase 1 (Current)
- ✅ Basic chat interface
- ✅ Conversation memory (in-memory)
- ✅ Provider abstraction (mock/OpenAI)
- ⏳ Database persistence (SQLite)

### Phase 2 (Planned)
- Long-term memory and user profiles
- Tool execution framework
- Web search integration
- Calendar and reminder tools
- Note-taking capabilities

### Phase 3 (Future)
- React frontend
- PostgreSQL migration
- Cloud deployment
- Multi-user support
- Advanced memory management

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

## 📝 License

[Your License Here]

## 🆘 Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Solution**: Ensure virtual environment is activated and dependencies are installed

**Issue**: Port already in use
- **Solution**: Change `PORT` in `.env` or specify a different port when running

**Issue**: OpenAI API errors
- **Solution**: Verify `OPENAI_API_KEY` is set correctly and you have API credits

---

**Built with ❤️ using FastAPI and Python**
