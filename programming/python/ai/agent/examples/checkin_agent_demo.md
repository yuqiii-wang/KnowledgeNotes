# Checkin App Agent Demo

This guide outlines the backend AI logic for an event check-in assistant. The goal is to handle a user query like: **"Hi, I am Alice, may i checkin?"**

## 1. System Architecture

To answer this question, the AI cannot simply hallucinate a response. It needs access to real-time data and policy documents. We will use a **ReAct (Reasoning + Acting)** approach.

### Key Components

1.  **User Context**: The system must know *who* is asking (e.g., `user_email: "alice@example.com"`).
2.  **Skills (The "Specialists")**: Modules that group specific tools and logic for distinct tasks (e.g., Registration vs. Security).
3.  **Tools (The "Hands")**: Python functions the AI can call to fetch structured data.
4.  **RAG / Knowledge Base (The "Library")**: Unstructured documents like the Event FAQ or Code of Conduct.
5.  **The Agent (The "Brain")**: The LLM loop that orchestrates the above.

## 2. Defining Tools & Skills

The agent needs tools to access information. Instead of connecting to a complex database for this demo, we will embed our "backend knowledge" directly in the Python code as dummy documents. The tools will simply "grep" (search) these documents.

### A. The Knowledge Source (Multi-Document RAG)

In a real-world scenario, you might have different rules for different events (e.g., a "Tech Conference" vs. a "Music Festival"). The AI needs to retrieve the correct policy document first.

> 💡 **Comparison with Traditional IT Dev**
>
> In traditional development, handling real-time updates (like **cancelling a performance due to bad weather**) often requires building a dedicated **Admin Dashboard** or database interface for operations staff.
>
> With an Agentic/RAG approach, this "Admin Backend" is often just the document itself. Operations staff can simply edit the text file (e.g., "Performance X is cancelled due to rain"), and the Agent immediately has access to the new information without any code changes or database migrations.

**Document 1: Tech Conference Policy**

```txt
TECH CONFERENCE 2024 - CHECK-IN PROTOCOL
----------------------------------------
1. ID REQUIREMENT: Government-issued photo ID required.
2. BAG POLICY: Laptops allowed. No large backpacks.
3. LATE ENTRY: Allowed up to 2 hours after start.
4. VIP ACCESS: Requires QR code scan + wristband.
```

**Document 2: Music Festival Policy**

```txt
SUMMER VIBES FESTIVAL - ENTRY RULES
-----------------------------------
1. ID REQUIREMENT: 21+ wristband check for alcohol areas. Ticket valid for entry.
2. BAG POLICY: Clear bags ONLY. No professional cameras.
3. RE-ENTRY: No re-entry allowed after 6 PM.
4. PROHIBITED: No outside food/drink, no umbrellas.
```

**Document 3: Registration Database**

```txt
REGISTRATION DATABASE
---------------------
1. Alice Smith (Tech Conf) - STATUS: CONFIRMED - TICKET: VIP
2. Bob Jones (Music Fest) - STATUS: CONFIRMED - TICKET: GA
```

### B. The Retrieval Tool (Finding the Right Doc via Embeddings)

Instead of hardcoding "if/else" rules, we use **Embeddings** to find the most relevant document. We convert the user's query (e.g., "Can I bring my laptop?") into a vector and compare it against our document vectors to find a match.

> 💡 In traditional code, you might write:  
> `if "laptop" in query.lower().strip(): return "Allowed"`  
> This is fragile. What if the user types "Lap top"? Or input has leading spaces? You'd need rigorous logic control (e.g., regex, ignorecase).  
> Furthermore, managing these rules usually requires a custom **Admin Backend**. With agents, the "Admin Backend" is just the text file itself. Updates are as simple as editing a doc.

```python
from typing import Annotated
import numpy as np

# Mocking an embedding function (in reality, you'd use OpenAI or HuggingFace)
def get_embedding(text):
    # Returns a random vector for demo purposes
    return np.random.rand(512)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class KnowledgeBaseTools:
    
    def __init__(self):
        # Index our documents with their embeddings
        self.doc_store = {
            "tech_conf": {"text": doc_tech_conf, "vector": get_embedding("tech conference business rules")},
            "music_fest": {"text": doc_music_fest, "vector": get_embedding("music festival party rules")}
        }

    @tool("get_relevant_policy")
    def get_relevant_policy(self, user_query: Annotated[str, "The user's specific question or event name"]):
        """
        Retrieves the most relevant policy document using Semantic Search (RAG).
        Use this when the user asks about rules but you don't know which document applies.
        """
        query_vector = get_embedding(user_query)
        best_score = -1
        best_doc = None
        
        # RAG Logic: Find the document with the highest similarity to the query
        for doc_id, data in self.doc_store.items():
            score = cosine_similarity(query_vector, data["vector"])
            if score > best_score:
                best_score = score
                best_doc = data["text"]
        
        # If the similarity is too low, we might return nothing (thresholding)
        if best_score < 0.5:
            return "No relevant policy found."
            
        return best_doc

    @tool("check_registration_doc")
    def check_registration_doc(self, query: Annotated[str, "The name or email to look for"]):
        """
        Searches the 'Attendee List' document for a person's record.
        Use this to find status and WHICH EVENT they are registered for.
        """
        # ... (Existing grep logic) ...
        pass
```

### C. Defining Multiple Skills (The "Brain" Modules)

Instead of one giant "Check-In" procedure, we split functionality into specialized **Skills**. The LLM acts as a **Router**: it looks at the "Description" of each skill to decide which one to load and use based on the user's first message.

#### 1. The Receptionist Skill (Registration Management)
Currently loaded when: "User asks about tickets, registration status, fees, or account details."

```markdown
---
name: Registration & Accounts
description: Use this skill when the user asks about tickets, registration status, fees, or account details.
---

# SKILL: Registration & Accounts

## GOAL

Manage attendee records and payment status.

## TOOLS

- `check_registration_doc(query)`: Search for a user's ticket.
- `check_event_capacity(event_name)`: Get current crowd level.
- `update_attendance_status(name, status)`: Update DB.
- `verify_email(email, code)`: Authentication.

## WORKFLOW

1.  **Status Checks**:
    - If asking about crowds/seats -> Call `check_event_capacity`.
    - If asking about personal ticket -> Call `check_registration_doc`.

2.  **Check-In Process**:
    - If user wants to check in -> Verify email -> Update status -> Hand off to Security.
```

#### 2. The Policy Expert Skill (RAG)

Currently loaded when: "User asks about rules, what to bring, code of conduct, or event times."

```markdown
---
name: Event Policy & Rules
description: Use this skill when the user asks about rules, what to bring, code of conduct, or event times.
---

# SKILL: Event Policy & Rules

## GOAL
Answer questions about event regulations using RAG.

## TOOLS
- `get_relevant_policy(query)`: Semantic search of rules.

## WORKFLOW
1. Identify the user's event (ask if unknown).
2. specific user query -> `get_relevant_policy`.
3. Summarize the rule clearly.
```

#### 3. The Security Officer Skill (Gatekeeper)
Currently loaded when: "User is ready to physically enter the venue or validate identity."

```markdown
---
name: Gate Security & Verification
description: Use this skill when the user is ready to physically enter the venue or validate identity.
---

# SKILL: Gate Security & Verification

## GOAL

Enforce physical entry requirements.

## TOOLS

- `verify_id_photo(photo, name)`
- `verify_location(lat, long)`
- `approve_attendance(name)` / `reject_attendance(reason)`

## WORKFLOW

1. Prerequisite: User must be "Checked In" by Registration Skill.
2. Perform ID Check.
3. Perform Location Check.
4. Grant/Deny Entry.
```

### D. The Router Logic (First Round)

When the conversation starts, the system prompt includes an **Index of Skills**. The system dynamically loads the `name` and `description` from the YAML frontmatter of each skill document.

**System Prompt:**
> You are the Event Master AI. You have access to the following skills. 
> carefully analyze the user's request and ACTIVATE the most relevant skill by calling `activate_skill(skill_name)`.
>
> **Available Skills:**
> 1.  **Registration & Accounts**: Use this skill when the user asks about tickets, registration status, fees, or account details.
> 2.  **Event Policy & Rules**: Use this skill when the user asks about rules, what to bring, code of conduct, or event times.
> 3.  **Gate Security & Verification**: Use this skill when the user is ready to physically enter the venue or validate identity.

**Example Conversation Trace:**
*   **User**: "Can I bring my dog?"
*   **AI (Router Match)**: "User is asking about rules -> Activate `policy_skill`."
*   **AI (Policy Skill)**: Calls `get_relevant_policy("pets")` -> Returns "No pets allowed."

---

### E. The Backend Tools (The "Hands")

The Python functions provide the raw capabilities referenced in the Skills above.

> **💡 Dev Tip: Documentation IS Code**
> When you register a tool, the AI framework reads your **Docstrings** and **Type Hints** to teach the LLM how to use it.
> *   The `docstring` tells the model **when** and **why** to use the tool.
> *   The `Annotated` type hints tell the model accurately **what arguments** to generate.

```python
class SecurityTools:
    
    @tool("verify_email")
    def verify_email(self, email: Annotated[str, "User's email address"], code: Annotated[str, "4-digit verification code"]):
        """
        Verifies ownership of an email address by checking a submitted code.
        (Simulated) Returns True if code is '1234'.
        """
        if code == "1234":
            return {"verified": True}
        return {"verified": False, "error": "Invalid code"}

    @tool("verify_id_photo")
    def verify_id_photo(self, photo_data: Annotated[str, "Simulated photo string"], name: str):
        """
        Analyzes a submitted ID photo to see if it matches the registered name.
        (Simulated) always returns True for demo.
        """
        return {"match": True, "confidence": 0.98}

    @tool("verify_location")
    def verify_location(self, lat: float, long: float):
        """
        Verifies if the user's phone is physically at the event venue.
        (Simulated) Checks if coordinates are within the venue geofence.
        """
        EVENT_LAT = 37.7749
        EVENT_LONG = -122.4194
        
        # Simple distance check (simulated)
        if abs(lat - EVENT_LAT) < 0.01 and abs(long - EVENT_LONG) < 0.01:
            return "inside_venue"
        return "outside_venue_perimeter"

    @tool("check_event_capacity")
    def check_event_capacity(self, event_name: str):
        """
        Returns the current crowd level and available seats.
        Use this to answer questions like 'is it crowded?' or 'are there tickets left?'.
        """
        # Simulated data
        return {
            "total_capacity": 500,
            "current_attendees": 450,
            "remaining_seats": 50,
            "status": "CROWDED"
        }

    @tool("update_attendance_status")
    def update_attendance_status(self, name: str, new_status: str):
        """
        Updates the attendance status in the database.
        Use this to mark a user as 'CHECKED_IN' or 'REFUSED_ENTRY'.
        """
        # In a real app, this would execute: UPDATE db SET status=? WHERE name=?
        return f"SUCCESS: Updated status for {name} to {new_status}."
    
    @tool("approve_attendance")
    def approve_attendance(self, name: str):
        """Grants access to the event."""
        return f"ACCESS GRANTED: Welcome, {name}!"

    @tool("reject_attendance")
    def reject_attendance(self, reason: str):
        """Denies access to the event."""
        return f"ACCESS DENIED: {reason}"
```

In this setup:
1.  **User asks**: "Can David Lee come in?"
2.  **LLM calls**: `check_registration_doc("David Lee")`
3.  **Tool returns**: `4. David Lee (david@example.com) - STATUS: CONFIRMED - TICKET: GA [FLAG: DUPLICATE]`
4.  **LLM analyzes**: "He is confirmed but flagged." -> Calls `get_event_policy("DUPLICATE")` (or appropriate RAG call)
5.  **Tool returns**: `3. FLAGS: If a registration is flagged as a 'DUPLICATE', the user may still attend...`
6.  **LLM actions**: Calls `update_attendance_status("David Lee", "CHECKED_IN")` then `approve_attendance("David Lee")`.

---

## 3. The Agent Workflow (Step-by-Step)

When the user asks **"Am I allowed to attend the event?"**, the backend logic follows this flow:

### Step 1: Input Processing & Intent Recognition
The LLM receives the prompt.
*   **System Prompt**: "You are a helpful check-in assistant. You verify eligibility based on registration status and event policies. You act on behalf of the event organizers."
*   **User Input**: "Am I allowed to attend?"
*   **Context**: `{ "current_user": "alice@example.com" }`

### Step 2: Reasoning (The "Thought")
The Agent analyzes the request. It realizes it cannot answer "Yes/No" without data.
*   *Internal Monologue*: "The user wants to know their attendance eligibility. First, I need to check if they have a valid ticket. I have the user's email from the context."

### Step 3: Tool Execution (The "Act")
The Agent selects the appropriate tool.
*   **Action**: Call `check_registration("alice@example.com")`
*   **Observation (Result)**:
    ```json
    {
      "status": "confirmed",
      "ticket_type": "general_admission",
      "flagged_for_review": true,
      "flag_reason": "potential_duplicate"
    }
    ```

### Step 4: Secondary Reasoning (Handling Complexity)
The Agent sees a flag ("potential_duplicate"). It might need to check policy.
*   *Internal Monologue*: "The user has a confirmed ticket, but it's flagged. I should check the event policy regarding duplicate registrations to see if this disqualifies them."

### Step 5: RAG Lookup (Optional but robust)
*   **Action**: Call `search_event_policy("duplicate registration policy")`
*   **Observation**: "Policy Section 4.2: Duplicate registrations solely for the purpose of holding spots are void. Accidental duplicates will be merged at check-in."

### Step 6: Final Synthesis
The Agent combines the tool output and the policy context to generate the answer.
*   **Final Answer**: "Yes, you have a confirmed General Admission ticket. However, your account is flagged as a potential duplicate. According to our policy, this usually just means we'll need to merge your records at the check-in desk, but you are allowed to attend."

> **💡 Dev Tip: The Conversation Loop**
> The "Agent" is actually just a `while` loop that keeps feeding **tool results** back into the LLM as new messages.
> 1. User Message -> 2. LLM Tool Call -> 3. Execute Python Code -> 4. Send Tool Result Message -> 5. LLM Final Answer

```python
def run_agent(user_query, user_context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User: {user_context['email']}\nQuery: {user_query}"}
    ]
    
    while True:
        # 1. Ask LLM what to do
        response = llm.chat(messages, tools=[check_registration, search_event_policy])
        
        # 2. Check if LLM wants to run a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # 3. Execute logic (Function Calling)
                function_name = tool_call.function.name
                args = tool_call.function.arguments
                
                print(f"🤖 Agent is calling {function_name} with {args}...")
                result = execute_tool(function_name, args)
                
                # 4. Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
        else:
            # 5. No more tools needed, return final answer
            return response.content
```
