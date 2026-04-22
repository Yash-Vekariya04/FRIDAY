from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

# Initialze the server
app = FastAPI()

# This is the exact endpoint our dashboard will connect to
@app.websocket("/ws/friday")
async def friday_audio_stream(websocket: WebSocket):

    # 1. Accept the incoming "Phone call" from the browser
    await websocket.accept()
    print("User dashboard is connected. F.R.I.D.A.Y. is listenning..." \
    "")
    try:
        # 2. The Infinite Listening Loop
        while True:
            # Wait for the chunk of the audio from the brouser
            audio_chunk = await websocket.receive_bytes()

            # 3. The Brain Processing (This is where Phase 3 & 4 code goes!)
                # transcribed_text = stt_model.process(audio_chunk)
                # ai_response = llm_brain.generate(transcribed_text)
                # audio_response = tts_model.speak(ai_response)

            # 4. Stream the response back to the browser instantly
            # We use await so the server doesn't freeze while sending
            await websocket.send_text("Processing command...")

            # In reality, you would send the raw audio bytes back here:
            # await websocket.send_bytes(audio_response)
    # If you close the browser tab on your phone, it safely handles the disconnect
    except WebSocketDisconnect:
        print("User disconnected. F.R.I.D.A.Y. returning to standby.")


# CHALLENGE

'''
Imagine you are driving in a car and talking to F.R.I.D.A.Y. via your phone. You drive through a short tunnel, and your 5G cellular signal drops out completely for just 2 seconds before coming right back.

With standard HTTP, a dropped signal just means a webpage fails to load, and you refresh it. But with a WebSocket, what happens to that permanent "phone call" connection when the 5G drops, and what must your JavaScript frontend be programmed to do immediately after you exit the tunnel?
'''
'''
MY:
the 2 packets of the audio will be send in which one will be before the 2 seconds gap and another will be after 2 seconds

GEMI:
That is a brilliant logical guess, and it is exactly how standard web traffic (HTTP) tries to behave by holding and retrying requests.

But with WebSockets, the reality is much more brutal.

Because a WebSocket is a persistent, live TCP connection, it behaves exactly like a real phone call. If your 5G drops completely, the connection doesn't just pause—it shatters. The connection is instantly severed. The audio packet you tried to send while inside the tunnel doesn't queue up; it just vanishes into the void, and the server registers a WebSocketDisconnect error.

Because of this, if you don't explicitly write code to handle it, F.R.I.D.A.Y. will just stay permanently disconnected until you manually refresh the web page.
'''