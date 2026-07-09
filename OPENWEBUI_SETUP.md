# Open WebUI Setup

## Option 1: Pipe Function (Recommended)

1. Open Open WebUI Admin Panel → Functions
2. Click "Create Function"
3. Paste the contents of `openclaw_pipe.py`
4. Set the name to "OpenClaw CEO"
5. Click Save
6. The "OpenClaw CEO" model will appear in the model dropdown

## Option 2: Direct API Connection

1. Open Open WebUI Admin Panel → Settings → Connections
2. Add a new OpenAI-compatible connection:
   - URL: `http://host.docker.internal:8765/v1`
   - Key: (leave blank or set OPENCLAW_API_KEY)
3. Save — models from OpenClaw will appear in the dropdown

## Configuration

The Pipe Function accepts these settings (Valves):
- `openclaw_url`: URL of the OpenClaw CEO server
- `poll_interval_seconds`: How often to check goal status (default: 5s)
- `max_poll_time_seconds`: Maximum wait time (default: 300s)

## Docker Compose

If running OpenClaw alongside Open WebUI:

```yaml
version: '3'
services:
  openclaw-ceo:
    build: ./openclaw-ceo
    ports:
      - "8765:8765"
    environment:
      NVIDIA_API_KEY: ${NVIDIA_API_KEY}
    volumes:
      - openclaw-data:/root/.openclaw

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      OPENAI_API_BASE_URL: http://openclaw-ceo:8765/v1
    volumes:
      - open-webui-data:/app/backend/data

volumes:
  openclaw-data:
  open-webui-data:
```
