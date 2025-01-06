import { useState, useEffect, useCallback, useRef } from 'react';

export const usePrediction = (gazeDataBuffer) => {
  const [predictions, setPredictions] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const lastPredictionTime = useRef(Date.now());

  const predictPreference = useCallback(async (data) => {
    if (!data || data.length === 0) return;
    
    try {
      setIsProcessing(true);
      console.log(`Sending prediction request with ${data.length} gaze points`);
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          gaze_data: data,
          model_name: 'lstm_method1',
          method: 'method1'
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('Prediction result:', result);
      
      setPredictions(result.predictions);
      setConfidence(result.confidence);
      setError(null);
      lastPredictionTime.current = Date.now();
      
    } catch (e) {
      console.error('Prediction error:', e);
      setError(e.message);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  useEffect(() => {
    // 1秒ごとに予測を実行
    const predictionInterval = setInterval(() => {
      if (!isProcessing && gazeDataBuffer && gazeDataBuffer.length > 0) {
        if (Date.now() - lastPredictionTime.current >= 1000) {
          predictPreference(gazeDataBuffer);
        }
      }
    }, 1000);

    return () => clearInterval(predictionInterval);
  }, [gazeDataBuffer, isProcessing, predictPreference]);

  return {
    predictions,
    confidence,
    isProcessing,
    error
  };
};

export default usePrediction;
