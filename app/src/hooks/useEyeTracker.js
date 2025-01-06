import { useState, useEffect, useCallback, useRef } from 'react';

const useEyeTracker = (defaultUrl = 'ws://localhost:8765') => {
  const [gazeData, setGazeData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);


  // Docker環境かどうかを確認し、適切なURLを設定
  const wsUrl = process.env.REACT_APP_EYE_TRACKER_URL || 
  process.env.EYE_TRACKER_URL || 
  'ws://host.docker.internal:8765';  // Docker環境用のデフォルト

  const connect = useCallback(() => {
    try {
      console.log(`Attempting to connect to WebSocket at ${wsUrl}`);
      
      if (socketRef.current) {
        socketRef.current.close();
      }

      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connection established');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'connection_status') {
            console.log(`[${new Date().toISOString()}] Connection status:`, data.status);
          } else {
            setGazeData(data);
          }
        } catch (err) {
          console.error('Error parsing gaze data:', err);
        }
      };

      ws.onclose = () => {
        console.log(`[${new Date().toISOString()}] WebSocket connection closed`);
        setIsConnected(false);
        
        // 再接続を試みる
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = setTimeout(connect, 5000);
      };

      ws.onerror = (error) => {
        console.error(`[${new Date().toISOString()}] WebSocket error:`, error);
        setError('Connection error');
      };

    } catch (err) {
      console.error('Connection error:', err);
      setError(`Connection error: ${err.message}`);
    }
  }, [wsUrl]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [connect]);

  return { gazeData, isConnected, error };
};

export default useEyeTracker;
