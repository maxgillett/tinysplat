import queue
import copy
import json
import base64
import asyncio
from uuid import uuid4

import numpy as np
import websockets
import numpy as np
import cv2
import torch
import torch.multiprocessing as mp 

from .scene import Camera

class Client:
    def __init__(self, session):
        self.ws = session
        self.camera = None

    async def send_image(self, img):
        # Encode image as base64, and send it as {"image: "base64data"}
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        retval, buffer = cv2.imencode('.jpg', img_bgr)
        jpg_as_text = base64.b64encode(buffer)
        await self.send_json({'image': jpg_as_text.decode('utf-8')})

    async def send_json(self, data):
        await self.ws.send(json.dumps(data))


class Viewer:
    def __init__(self, scene, ip="127.0.0.1", port=8765, device="cuda:0"):
        self.device = device
        self.port = port
        self.server = None
        self.clients = set()
        self.scene = scene
        self.async_queue = asyncio.Queue(maxsize=1)

    async def handle_client(self, websocket, path):
        client = Client(websocket)
        self.clients.add(client)
        try:
            async for message in websocket:
                await self.handle_message(client, message)
        finally:
            self.clients.remove(websocket)

    async def run(self):
        self.server = await websockets.serve(self.handle_client, "localhost", self.port)
        asyncio.create_task(self.process_async_queue())
        await self.server.wait_closed()

    async def handle_message(self, client, message):
        print("Got message: ", message)
        msg = json.loads(message)

        if msg["type"] == "cameraInfo":
            # Duplicate a camera from the scene, unless a new one is provided
            camera = copy.copy(self.scene.cameras[0])
            camera.rescale(4)
            client.camera = camera
            await self.async_queue.put((client, msg))
        elif msg["type"] == "renderRequest":
            # Push a request to the queue, evicting previous one if necessary
            if self.async_queue.full():
                _ = await self.async_queue.get()
            await self.async_queue.put((client, msg))

    async def process_async_queue(self):
        print("Processing async queue")
        while True:
            client, msg = await self.async_queue.get()

            # Update the view matrix
            position = torch.as_tensor(msg["position"], dtype=torch.float32)
            quat = np.asarray(msg["quat"], dtype=np.float32)
            client.camera.update_view_matrix(position, quat)

            # Render a frame
            with torch.no_grad():
                img = self.scene.render(client.camera).detach().cpu().numpy()
            img = img * 255

            # Render at 10fps. This should be tuned to network latency
            await client.send_image(img)
            await asyncio.sleep(0.1)

    def stop(self):
        self.server.close()
