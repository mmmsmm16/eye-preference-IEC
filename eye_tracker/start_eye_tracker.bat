@echo off
call activate eye_tracker
python src/websocket_server.py
pause
