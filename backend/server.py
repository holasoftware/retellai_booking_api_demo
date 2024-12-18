import json
import os
import asyncio
import logging

import httpx

from dotenv import load_dotenv

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from concurrent.futures import TimeoutError as ConnectionTimeoutError

from retell import Retell

from .custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
from .llm import LlmClient  # or use .llm_with_func_calling


CREATE_WEB_CALL_RETELLAI_ENDPOINT = 'https://api.retellai.com/v2/create-web-call'

load_dotenv(override=True)
RETELL_API_KEY = os.environ["RETELL_API_KEY"]


app = FastAPI()
retell = Retell(api_key=RETELL_API_KEY)

logger = logging.getLogger(__name__)


class WebCallRequest(BaseModel):
    agent_id: str
    metadata: dict | None = None
    retell_llm_dynamic_variables: dict | None = None


# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=RETELL_API_KEY,
            signature=str(request.headers.get("X-Retell-Signature")),
        )
        if not valid_signature:
            logger.warning(
                "Received Unauthorized",
                post_data["event"],
                post_data["data"]["call_id"],
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})

        event_type = post_data["event"]

        if event_type == "call_started":
            logger.info("Call started event %s", post_data["data"]["call_id"])
        elif event_type == "call_ended":
            logger.info("Call ended event %s", post_data["data"]["call_id"])
        elif event_type == "call_analyzed":
            logger.info("Call analyzed event %s", post_data["data"]["call_id"])
        else:
            logger.info("Unknown event", event_type)
        return JSONResponse(status_code=200, content={"received": True})
    except Exception as err:
        logger.error(f"Error in webhook: {err}")
        return JSONResponse(
            status_code=500, content={"message": "Internal Server Error"}
        )


@app.post("/create-web-call")
async def create_web_call(web_call_request: WebCallRequest):
    # Prepare the payload for the API request
    payload = {
        "agent_id": web_call_request.agent_id
    }

    # Conditionally add optional fields if they are provided
    if web_call_request.metadata:
        payload["metadata"] = web_call_request.metadata

    if web_call_request.retell_llm_dynamic_variables:
        payload["retell_llm_dynamic_variables"] = web_call_request.retell_llm_dynamic_variables

    try:
        response = await httpx.post(
            CREATE_WEB_CALL_RETELLAI_ENDPOINT,
            data=payload,
            headers={
                'Authorization': 'Bearer %s' % RETELL_API_KEY,
                'Content-Type': 'application/json'
            }
        )
    except httpx.HTTPError as exc:
        logger.exception('Error creating web call');
        raise HTTPException(status_code=500, detail={ "error": 'Failed to create web call' });
    else:
        data = response.json()["data"]
        return JSONResponse(status_code=201, content=data)


# Start a websocket server to exchange text input and output with Retell server. Retell server
# will send over transcriptions and other information. This server here will be responsible for
# generating responses with LLM and send back to Retell server.
@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    try:
        await websocket.accept()
        llm_client = LlmClient()

        # Send optional config to Retell server
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,
                "call_details": True,
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)

        # Send first message to signal ready of server
        response_id = 0
        first_event = llm_client.draft_begin_message()
        await websocket.send_json(first_event.__dict__)
        interaction_type = request_json["interaction_type"]


        async def handle_message(request_json):
            nonlocal response_id

            # There are 5 types of interaction_type: call_details, pingpong, update_only, response_required, and reminder_required.
            # Not all of them need to be handled, only response_required and reminder_required.
            if interaction_type == "call_details":
                print(json.dumps(request_json, indent=2))
                return
            if interaction_type == "ping_pong":
                await websocket.send_json(
                    {
                        "response_type": "ping_pong",
                        "timestamp": request_json["timestamp"],
                    }
                )
                return
            if interaction_type == "update_only":
                return
            if (
                interaction_type == "response_required"
                or interaction_type == "reminder_required"
            ):
                response_id = request_json["response_id"]
                request = ResponseRequiredRequest(
                    interaction_type=request_json["interaction_type"],
                    response_id=response_id,
                    transcript=request_json["transcript"],
                )
                print(
                    f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                )

                async for event in llm_client.draft_response(request):
                    await websocket.send_json(event.__dict__)
                    if request.response_id < response_id:
                        break  # new response needed, abandon this one

        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        logger.info(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        logger.info("Connection timeout for {call_id}")
    except Exception as e:
        logger.error(f"Error in LLM WebSocket: {e} for {call_id}")
        await websocket.close(1011, "Server error")
    finally:
        logger.info(f"LLM WebSocket connection closed for {call_id}")