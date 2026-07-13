# AI Team Architecture — CEO-Centric

## Overview

OpenHands UI is the front-end. CEO (`localhost:8282`) is the back-end for ALL conversations.
Every new conversation in OpenHands UI hits CEO first. CEO classifies, routes, and orchestrates.

```
OpenHands UI (port 3000)
       │
       │ base_url = http://localhost:8282/v1
       │
       ▼
CEO (port 8282) ─── intent classifier ─── mode dispatch
       │                  │
       │    ┌─────────────┼──────────────┐
       ▼    ▼             ▼              ▼
      Ask  Plan          Code          Auto
       │     │            │             │
       │     │    ┌───────┼───────┐     │
       │     │   Code  Debug  Quick  ───┘
       │     │   Inline Batch          (escalates)
       ▼     ▼    ▼       ▼
     NVIDIA Nemo 550B
```

## Components

### CEO (`localhost:8282`)

The LLM endpoint. Implements OpenAI-compatible `/v1/chat/completions`.
OpenHands UI sets `base_url` to CEO. All requests go through CEO.

**Existing assets:**
- Chat router (`/app/routers/chat.py`) — `/v1/chat/completions` proxy
- Goal service — task decomposition engine
- Agent manager — sub-agent lifecycle tracking
- Kanban dashboard — visual task board

**New components needed:**
1. Intent classifier — reads message, decides mode
2. Mode router — strips/adds tool defs, swaps system prompts
3. Plan gate — proposes plan → waits for confirmation → executes
4. Execution bridge — dispatches Code sub-tasks to OpenHands agent servers

### OpenHands UI (port 3000)

Front-end only. Each conversation has a "mode" that CEO manages.
OpenHands sends tool definitions to CEO. CEO strips them for Ask/Plan, passes them through for Code.

## Modes

### Ask — Pure Q&A

```
User: "What is JWT?"
       │
CEO strips tool definitions from request
       │
CEO adds system prompt: "You are a helpful assistant. Answer concisely. Do not use tools."
       │
CEO calls NVIDIA Nemo 550B
       │
CEO returns text response
       │
OpenHands displays it (no tool calls)
```

- No tools. No agent loop. Just the LLM.
- CEO strips tool definitions from the incoming request before forwarding.
- System prompt forces Q&A-only behavior.

### Plan — Task Decomposition

```
User: "Build a REST API for auth"
       │
CEO strips tool definitions
       │
CEO calls NVIDIA with planning system prompt
       │
CEO returns structured plan (steps, dependencies, effort estimate)
       │
OpenHands displays the plan
       │
[User reviews and decides]
```

- CEO does NOT execute. It decomposes and proposes.
- Plan is returned as structured text (steps, files affected, dependencies).
- User reviews the plan in the same conversation thread.
- If user says "execute", the conversation switches to Code mode.

**Plan format:**

```
## Plan: "Build a REST API for auth"

### Step 1: Database schema
- Create `users` table (email, password_hash, created_at)
- Files: src/db/migrations/001_users.sql

### Step 2: Auth routes
- POST /register, POST /login, POST /refresh
- Files: src/routes/auth.py

### Step 3: JWT middleware
- Token generation, verification, expiry
- Files: src/middleware/auth.py

### Dependencies: Step 1 → Step 2 → Step 3
```

### Code — Full Agent with Tools

```
User: "Add error handling to auth.py"
       │
CEO passes through the FULL request (messages + tool definitions)
       │
CEO proxies to NVIDIA Nemo 550B unchanged
       │
NVIDIA responds with tool calls OR text
       │
CEO returns response unchanged
       │
OpenHands agent executes tool calls (file_editor, terminal, etc.)
```

Code mode has 5 sub-modes that modify agent behavior:

| Sub-mode | Behavior |
|---|---|
| **Code** | Full OpenHands agent with all tools. General coding. |
| **Debug** | Error trace in → reproduce → diagnose → fix → verify. Uses terminal + file tools. |
| **Quick** | Trivial one-shot edit. No analysis, no exploration. Just "change X to Y". Agent runs minimal loop. |
| **Inline** | Reads file around cursor position, makes targeted suggestion. Agent only uses file_editor. |
| **Batch** | Takes a spec file or list of prompts. Runs each as a separate agent task. Returns results per item. |

**Sub-mode selection:** CEO classifies the user's intent within Code mode. "this is broken" → Debug. "change the color" → Quick. "fix this line" → Inline. "do all of these" → Batch. Otherwise → Code.

### Auto — Self-Routing

```
User: "I need a login system"
       │
CEO classifies intent
       │
├─ Simple Q&A ? ───→ Ask mode (direct answer)
│
└─ Complex task ? ──→ Plan mode

       CEO proposes plan
              │
User: "looks good, execute"
       │
CEO detects confirmation
       │
CEO switches conversation to Code mode
       │
Subsequent messages go through Code
```

Auto mode is the default. It starts as Ask. If the classifier detects a complex task, it escalates to Plan. After user confirms, it enters Code. The conversation thread stays the same — user sees the Q&A, then the plan, then the execution, all in one chat.

**Confirmation detection:** CEO watches for phrases like "execute", "go ahead", "start", "looks good" in context of a plan being present. Simple pattern match.

## Mode Dispatch Table

| Incoming Request | Mode (explicit) | Auto Classification | CEO Action |
|---|---|---|---|
| tools in request | Code | — | Pass through to NVIDIA |
| system prompt says "ask" | Ask | — | Strip tools, use Q&A prompt |
| "explain X" | — | Ask | Strip tools, Q&A prompt |
| "build X" | — | Plan | Strip tools, planning prompt |
| "fix error" | — | Code | Pass through with tools |
| "change line 42" | — | Code → Quick | Pass through with minimal agent |
| "run these 10 tasks" | — | Code → Batch | Split into sub-agent tasks |
| Plan exists + "execute" | — | Plan → Code | Pass through with tools |
| Plan exists + "no" | — | Plan → Ask | Strip tools, back to Q&A |

## Conversation Flow (Auto Mode Example)

```
[1] User: "I need to build a user authentication system"

    CEO: [classifies → Plan]
    CEO: "Here's my plan:
         Step 1: Create users table (schema migration)
         Step 2: Build registration/login routes
         Step 3: JWT middleware
         Should I proceed?"

[2] User: "Yes, start with step 1"

    CEO: [detects confirmation → Code mode]
    CEO: [passes through to NVIDIA with tools]
    OpenHands: [creates migration file, executes it]

[3] OpenHands: "Created users table. Ready for step 2."

    User: "Go ahead"

    CEO: [Code mode continues]
    ... (same thread, same conversation)
```

## Implementation Order

1. **Mode classifier** — lightweight prompt to classify intent from the user's first message
2. **Ask mode** — strip tools, add Q&A system prompt, proxy to NVIDIA
3. **Plan mode** — planning system prompt, structured plan output, confirmation gate
4. **Code mode** — pass-through proxy (already partially exists in chat.py)
5. **Code sub-modes** — Quick, Debug, Inline, Batch (agent-side behavior changes)
6. **Auto mode** — wire classifier → Plan gate → Code execution
7. **OpenHands config** — set `base_url` to CEO, add mode profiles
