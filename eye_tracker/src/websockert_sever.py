import asyncio
import json
import websockets
import tobii_research as tr
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EyeTrackerServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.eye_tracker = None
        self.active_connections = set()
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.server = None

    def find_eyetracker(self):
        """アイトラッカーを検出"""
        try:
            eye_trackers = tr.find_all_eyetrackers()
            if len(eye_trackers) == 0:
                logger.error("No eye trackers found")
                return None
            logger.info(f"Found eye tracker: {eye_trackers[0].model}")
            return eye_trackers[0]
        except Exception as e:
            logger.error(f"Error finding eye tracker: {str(e)}")
            return None

    def gaze_data_callback(self, gaze_data):
        """視線データのコールバック関数"""
        try:
            if not self.active_connections:
                return

            data = {
                'timestamp': gaze_data['device_time_stamp'],
                'left_x': gaze_data['left_gaze_point_on_display_area'][0],
                'left_y': gaze_data['left_gaze_point_on_display_area'][1],
                'right_x': gaze_data['right_gaze_point_on_display_area'][0],
                'right_y': gaze_data['right_gaze_point_on_display_area'][1]
            }

            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast(json.dumps(data)))
                )
        except Exception as e:
            logger.error(f"Error in gaze_data_callback: {str(e)}")

    async def broadcast(self, message):
        """全クライアントにメッセージをブロードキャスト"""
        if not self.active_connections:
            return

        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                disconnected.add(websocket)

        self.active_connections.difference_update(disconnected)

    async def handle_websocket(self, websocket):
        """個別のWebSocket接続を処理"""
        try:
            logger.info(f"New client connected from {websocket.remote_address}")
            
            # クライアントの検証やハンドシェイクの処理をここで行うことができます
            self.active_connections.add(websocket)
            
            # 接続確認メッセージを送信
            await websocket.send(json.dumps({
                "type": "connection_status",
                "status": "connected",
                "message": "Successfully connected to eye tracker server"
            }))

            # クライアントからのメッセージを待機
            async for message in websocket:
                try:
                    data = json.loads(message)
                    logger.info(f"Received message: {data}")
                    # 必要に応じてメッセージを処理
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed normally")
        except Exception as e:
            logger.error(f"Error handling client: {str(e)}")
        finally:
            self.active_connections.remove(websocket)
            logger.info("Client disconnected")

    async def start(self):
        """サーバーを開始"""
        try:
            self.loop = asyncio.get_running_loop()
            
            # アイトラッカーの初期化
            self.eye_tracker = self.find_eyetracker()
            if not self.eye_tracker:
                raise Exception("No eye tracker found")

            # 視線データの購読開始
            self.eye_tracker.subscribe_to(
                tr.EYETRACKER_GAZE_DATA,
                self.gaze_data_callback,
                as_dictionary=True
            )

            # WebSocketサーバーの設定
            logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
            async with websockets.serve(
                self.handle_websocket,
                self.host,
                self.port,
                ping_interval=20,  # Keep-alive pingを20秒ごとに送信
                ping_timeout=60    # ping timeoutを60秒に設定
            ) as server:
                self.server = server
                logger.info("WebSocket server is running")
                await asyncio.Future()  # サーバーを永続的に実行

        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            raise
        finally:
            if self.eye_tracker:
                self.eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
            logger.info("Server shutdown complete")

    async def cleanup(self):
        """リソースのクリーンアップ"""
        if self.eye_tracker:
            self.eye_tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, self.gaze_data_callback)
        
        for connection in self.active_connections:
            await connection.close()
        
        self.executor.shutdown(wait=True)
        logger.info("Cleanup completed")

def main():
    # localhost の代わりに 0.0.0.0 を使用して全てのインターフェースでリッスン
    server = EyeTrackerServer(host="0.0.0.0", port=8765)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")
        asyncio.run(server.cleanup())
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        asyncio.run(server.cleanup())

if __name__ == "__main__":
    main()
