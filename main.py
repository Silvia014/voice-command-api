import json
import os
from typing import Optional
from fastapi import Request

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel


load_dotenv()

app = FastAPI(title="Voice Command API")


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# In-memory storage
# --------------------------------------------------

tasks = []


# --------------------------------------------------
# Models
# --------------------------------------------------

class TaskCreate(BaseModel):
    title: str
    done: bool = False


class TaskUpdate(BaseModel):
    title: str
    done: bool


class TaskPatch(BaseModel):
    title: Optional[str] = None
    done: Optional[bool] = None


class Instruction(BaseModel):
    transcription: str


# --------------------------------------------------
# Groq
# --------------------------------------------------

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# --------------------------------------------------
# TASK ENDPOINTS
# --------------------------------------------------

@app.get("/tasks")
def get_tasks():
    return tasks


@app.post("/tasks")
def create_task(task: TaskCreate):
    new_id = max(
        [item["id"] for item in tasks],
        default=0
    ) + 1

    new_task = {
        "id": new_id,
        "title": task.title,
        "done": task.done,
    }

    tasks.append(new_task)

    return new_task


@app.put("/tasks/{task_id}")
def replace_task(task_id: int, task: TaskUpdate):

    for index, existing_task in enumerate(tasks):

        if existing_task["id"] == task_id:

            updated_task = {
                "id": task_id,
                "title": task.title,
                "done": task.done,
            }

            tasks[index] = updated_task

            return updated_task

    raise HTTPException(
        status_code=404,
        detail="Task not found"
    )


@app.patch("/tasks/{task_id}")
def update_task(task_id: int, task: TaskPatch):

    for existing_task in tasks:

        if existing_task["id"] == task_id:

            if task.title is not None:
                existing_task["title"] = task.title

            if task.done is not None:
                existing_task["done"] = task.done

            return existing_task

    raise HTTPException(
        status_code=404,
        detail="Task not found"
    )


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):

    for index, task in enumerate(tasks):

        if task["id"] == task_id:

            tasks.pop(index)

            return {
                "message": "Task deleted successfully"
            }

    raise HTTPException(
        status_code=404,
        detail="Task not found"
    )


# --------------------------------------------------
# GROQ: INTERPRET INSTRUCTION
# --------------------------------------------------

def interpret_instruction(transcription: str):

    system_prompt = """
You are an API command parser for a task management application.

Convert the user's natural-language task command into exactly one
API operation.

You MUST respond with ONLY valid JSON.

Do NOT write explanations.
Do NOT use markdown.
Do NOT use code fences.
Do NOT return any text outside the JSON object.

The JSON format is:

{
    "endpoint": "/tasks",
    "method": "POST",
    "params": {}
}

Allowed operations:

GET /tasks
POST /tasks
PUT /tasks/{task_id}
PATCH /tasks/{task_id}
DELETE /tasks/{task_id}

Examples:

"añade comprar leche a mi lista"

{
    "endpoint": "/tasks",
    "method": "POST",
    "params": {
        "title": "Comprar leche"
    }
}

"muéstrame mis tareas"

{
    "endpoint": "/tasks",
    "method": "GET",
    "params": {}
}

"marca la tarea 1 como completada"

{
    "endpoint": "/tasks/1",
    "method": "PATCH",
    "params": {
        "done": true
    }
}

"cambia la tarea 1 a comprar pan"

{
    "endpoint": "/tasks/1",
    "method": "PATCH",
    "params": {
        "title": "Comprar pan"
    }
}

"elimina la tarea 1"

{
    "endpoint": "/tasks/1",
    "method": "DELETE",
    "params": {}
}
"""

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": transcription,
                },
            ],
            response_format={
                "type": "json_object"
            },
        )

        content = completion.choices[0].message.content

        return json.loads(content)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error interpreting instruction: {str(error)}"
        )


# --------------------------------------------------
# EXECUTE ROUTED TASK OPERATION
# --------------------------------------------------

def execute_instruction(instruction):

    endpoint = instruction.get("endpoint")
    method = instruction.get("method", "").upper()
    params = instruction.get("params", {})

    # ----------------------------
    # GET /tasks
    # ----------------------------

    if endpoint == "/tasks" and method == "GET":
        return get_tasks()

    # ----------------------------
    # POST /tasks
    # ----------------------------

    if endpoint == "/tasks" and method == "POST":
        task = TaskCreate(**params)
        return create_task(task)

    # ----------------------------
    # Extract task ID
    # ----------------------------

    if endpoint.startswith("/tasks/"):

        try:
            task_id = int(endpoint.split("/")[-1])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid task ID"
            )

        # ------------------------
        # PUT
        # ------------------------

        if method == "PUT":
            task = TaskUpdate(**params)
            return replace_task(task_id, task)

        # ------------------------
        # PATCH
        # ------------------------

        if method == "PATCH":
            task = TaskPatch(**params)
            return update_task(task_id, task)

        # ------------------------
        # DELETE
        # ------------------------

        if method == "DELETE":
            return delete_task(task_id)

    raise HTTPException(
        status_code=400,
        detail="Unsupported instruction"
    )


# --------------------------------------------------
# POST /instruction
# --------------------------------------------------

@app.post("/instruction")
def process_instruction(instruction: Instruction):

    parsed_instruction = interpret_instruction(
        instruction.transcription
    )

    return parsed_instruction


# --------------------------------------------------
# POST /transcribe
# --------------------------------------------------

@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
    transcription: Optional[str] = Form(None),
):
    content_type = request.headers.get("content-type", "")

    # --------------------------------------------------
    # JSON manual transcription
    # --------------------------------------------------

    if "application/json" in content_type:

        body = await request.json()

        transcription = body.get("transcription")

        if not transcription or not transcription.strip():
            raise HTTPException(
                status_code=400,
                detail="No valid transcription was received."
            )

        transcription = transcription.strip()

    # --------------------------------------------------
    # Audio transcription
    # --------------------------------------------------

    elif file is not None:

        try:

            audio_bytes = await file.read()

            transcription_response = groq_client.audio.transcriptions.create(
                file=(
                    file.filename or "audio.webm",
                    audio_bytes,
                ),
                model="whisper-large-v3",
                language=language if language else None,
            )

            transcription = transcription_response.text.strip()

        except Exception as error:

            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(error)}"
            )

    else:

        raise HTTPException(
            status_code=400,
            detail="Provide an audio file or a transcription."
        )

    # --------------------------------------------------
    # Interpret command
    # --------------------------------------------------

    instruction = interpret_instruction(
        transcription
    )

    # --------------------------------------------------
    # Execute task operation
    # --------------------------------------------------

    result = execute_instruction(
        instruction
    )

    return {
        "transcription": transcription,
        "instruction": instruction,
        "result": result,
    }