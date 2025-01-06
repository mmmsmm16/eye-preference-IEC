import asyncio
import websockets
import json
import tobii_research as tr
import time
from queue import Queue
from threading import Thread

# データキュー
gaze_data_queue = Queue()

def gaze_data_callback(gaze_data):
    left_eye = gaze_data['left_gaze_point_on_display_area']
    right_eye = gaze_data['right_gaze_point_on_display_area']
    data = {
        'timestamp': gaze_data['device_time_stamp'],
        'left_x': left_eye[0],
        'left_y': left_eye[1],
        'right_x': right_eye[0],
        'right_y': right_eye[1]
    }
    gaze_data_queue.put(data)

def tobii_data_collection():
    eyetrackers = tr.find_all_eyetrackers()
    if len(eyetrackers) == 0:
        print("No eye trackers found")
        return
    
    eyetracker = eyetrackers[0]
    print(f"Found eye tracker: {eyetracker.model}")
    
    eyetracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    
    print("Collecting gaze data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)  # スリープを短くして CPU 使用率を下げる
    except KeyboardInterrupt:
        pass
    finally:
        eyetracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
        print("Stopped gaze data collection.")

async def websocket_handler(websocket, path):
    print(f"Client connected from {websocket.remote_address}")
    try:
        while True:
            if not gaze_data_queue.empty():
                data = gaze_data_queue.get()
                await websocket.send(json.dumps(data))
            else:
                await asyncio.sleep(0.001)  # 短い待機時間
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    # Tobiiデータ収集を別スレッドで開始
    tobii_thread = Thread(target=tobii_data_collection)
    tobii_thread.start()

    # WebSocketサーバーを開始
    server = await websockets.serve(websocket_handler, "0.0.0.0", 8765)
    print("WebSocket server started on ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
