import React, { useState, useEffect, useCallback } from 'react';
import { Container, Box, Typography, LinearProgress, Button } from '@mui/material';  // Buttonを追加
import useEyeTracker from '../hooks/useEyeTracker';
import ImageGallery from './ImageGallery';
import ResultDisplay from './ResultDisplay';
import SelectionDisplay from './SelectionDisplay';
import SelectionConfirmation from './SelectionConfirmation';
import sessionManager from '../utils/sessionManager';
import dataManager from '../utils/dataManager';
import predictionService from '../services/predictionService';
import StartScreen from './StartScreen';
import imageGenerationService from '../services/imageGenerationService';
import FinalResultDisplay from './FinalResultDisplay';  

// アプリケーションのフェーズを定義
const PHASES = {
  SELECTION: 'selection',
  CHOOSING: 'choosing',
  PROCESSING: 'processing',
  CONFIRMING: 'confirming',
  FINAL_RESULT: 'final_result'  // 新しいフェーズ
};

const SPACE_KEY = 32;
const ENTER_KEY = 13;
const DEBUG_KEY = 68;

const ERROR_MESSAGES = {
  PREDICTION_FAILED: 'Prediction failed. Please try again.',
  SAVE_FAILED: 'Failed to save data. Please try again.',
  CONNECTION_ERROR: 'Connection error. Please check your connection.',
  GENERATION_FAILED: 'Failed to generate images. Please try again.'
};

// 利用可能なモデルの定義
const AVAILABLE_MODELS = [
  { id: 'lstm_method1', name: 'LSTM Method 1', type: 'lstm' },
  { id: 'lstm_method2', name: 'LSTM Method 2', type: 'lstm' },
  { id: 'transformer_method1', name: 'Transformer Method 1', type: 'transformer' },
  { id: 'transformer_method2', name: 'Transformer Method 2', type: 'transformer' }
];

function App() {
  // 基本状態の管理
  const [isDebugMode, setIsDebugMode] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [phase, setPhase] = useState(PHASES.SELECTION);
  const [images, setImages] = useState([]);
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [predictionConfidence, setPredictionConfidence] = useState(null);
  const [finalSelectedImage, setFinalSelectedImage] = useState(null);
  const [sessionStartTime, setSessionStartTime] = useState(null);
  
  // アプリケーション設定の状態
  const [interactionMode, setInteractionMode] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState(null);

  // エラー関連の状態
  const [showErrorDialog, setShowErrorDialog] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showConfirmEnd, setShowConfirmEnd] = useState(false);

  const [fixedPlacement, setFixedPlacement] = useState(true);  // 配置設定の状態を追加
  const [previousSelectedImage, setPreviousSelectedImage] = useState(null);  // 前回選択された画像を保持  

  // アイトラッカーの接続
  const { gazeData, isConnected: isEyeTrackerConnected } = useEyeTracker('ws://host.docker.internal:8765');

const formatJapanTime = (date) => {
    return new Date(date).toLocaleString('ja-JP', {
      timeZone: 'Asia/Tokyo',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };
  
  const calculateDurationInSeconds = (startTime, endTime) => {
    return Math.round((new Date(endTime) - new Date(startTime)) / 1000);
  };

  // セッション開始の処理
const handleStartSession = useCallback(async ({
    interactionMode: mode,
    selectedModel: model,
    prompt: inputPrompt,
    negativePrompt: inputNegativePrompt,
    fixedPlacement: placement,
  }) => {
    try {
      const startTime = new Date().toISOString();
      setSessionStartTime(startTime);
      setInteractionMode(mode);
      sessionManager.setInteractionMode(mode);
      setSelectedModel(model);
      setPrompt(inputPrompt);
      setNegativePrompt(inputNegativePrompt);
      setFixedPlacement(placement);
      
      const sessionId = await sessionManager.startNewSession();
      
      // セッションメタデータを保存（開始時刻を含む）
      const metadata = {
        evaluationMethod: mode,
        modelId: model || null,
        prompt: inputPrompt,
        negativePrompt: inputNegativePrompt || null,
        fixedPlacement: placement,
        eyeTrackerConnected: isEyeTrackerConnected,
        timestamp: startTime,
        startTime: formatJapanTime(startTime),  // 日本時間でフォーマット
        endTime: null,
        sessionDurationSeconds: null,  // 秒単位に変更
        finalSelectedImage: null
      };

      console.log('Saving session metadata:', metadata); // デバッグログ追加
      
      await dataManager.saveSessionMetadata(sessionId, metadata);
      
      setSessionStarted(true);
      setCurrentStep(0);
      setPreviousSelectedImage(null);
      
      if (isEyeTrackerConnected) {
        dataManager.startCollection();
      }
      
      setPhase(PHASES.PROCESSING);
      const result = await imageGenerationService.generateImages(
        inputPrompt,
        inputNegativePrompt,
        null,
        0,
        4
      );
  
      if (result && result.images) {
        setImages(result.images.map(img => ({
          ...img,
          src: imageGenerationService.getImageUrl(
            sessionId,
            0,
            img.url.split('/').pop()
          )
        })));
        setPhase(PHASES.SELECTION);
      }
    } catch (error) {
      console.error('Error starting session:', error);
      setErrorMessage('Failed to start session');
      setShowErrorDialog(true);
      setSessionStarted(false);
    }
  }, [isEyeTrackerConnected]);
  

  // キーボードイベントの処理
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (phase === PHASES.SELECTION) {  // 評価モードの条件を削除
        if (event.keyCode === SPACE_KEY) {
          handleSpacePress();
        } else if (event.keyCode === ENTER_KEY && interactionMode === 'gaze') {
          handleEnterPress();
        }
      } else if (event.keyCode === DEBUG_KEY && event.ctrlKey) {
        setIsDebugMode(prev => !prev);
      }
    };
  
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [phase, interactionMode]);

  // 視線データの処理
  useEffect(() => {
    // セッションが開始されていない、またはアイトラッカーが接続されていない場合は収集しない
    if (!sessionStarted || !isEyeTrackerConnected || !gazeData) return;
    
    // SELECTION フェーズの時のみデータを収集
    if (phase === PHASES.SELECTION) {
      dataManager.addGazeData(gazeData);
    }
  }, [sessionStarted, phase, gazeData, isEyeTrackerConnected]);

  // スペースキー押下時の処理
  const handleSpacePress = useCallback(async () => {
    if (phase !== PHASES.SELECTION) return;
  
    // 視線評価モードの場合
    if (interactionMode === 'gaze') {
      setPhase(PHASES.PROCESSING);
      dataManager.stopCollection();
      const gazeBuffer = dataManager.clearGazeBuffer();
      
      if (!gazeBuffer || gazeBuffer.length < 10) {
        setErrorMessage('Not enough gaze data collected');
        setShowErrorDialog(true);
        return;
      }
    
      sessionManager.setCurrentGazeData(gazeBuffer);
      
      try {
        // 予測を実行
        const result = await predictionService.getPrediction(gazeBuffer);
        setPrediction(result.predictedIndex);
        setPredictionConfidence(result.confidence);
        setSelectedImage(result.predictedIndex);
    
        // 予測結果に基づいて直接次のステップへ進む
        const stepData = {
          images: images,
          prompt,
          generation: currentStep,
          timestamp: new Date().toISOString(),
          selectedImage: result.predictedIndex,
          prediction: result.predictedIndex,
          predictionConfidence: result.confidence
        };
    
        // データを保存
        await dataManager.saveStepData(
          sessionManager.sessionId,
          currentStep,
          gazeBuffer,
          stepData
        );
    
        // 次の画像を生成
        const selectedImageData = images[result.predictedIndex];
        const nextResult = await imageGenerationService.generateImages(
          prompt,
          negativePrompt,
          selectedImageData.latent_vector,
          currentStep + 1,
          4
        );
    
        if (nextResult && nextResult.images) {
          // 生成された画像を表示用に処理
          const newImages = nextResult.images.map(img => ({
            ...img,
            src: imageGenerationService.getImageUrl(
              sessionManager.sessionId,
              currentStep + 1,
              img.url.split('/').pop()
            )
          }));
    
          // 配置に従って画像を並べ替え
          let displayImages;
          if (fixedPlacement) {
            displayImages = newImages;
          } else {
            const selectedImage = newImages[0];
            const otherImages = newImages.slice(1);
            const insertIndex = Math.floor(Math.random() * 4);
            displayImages = [...otherImages];
            displayImages.splice(insertIndex, 0, selectedImage);
          }
    
          setImages(displayImages);
          setCurrentStep(prev => prev + 1);
          setPhase(PHASES.SELECTION);
    
          // 視線データの収集を再開
          if (isEyeTrackerConnected) {
            dataManager.startCollection();
          }
        }
      } catch (error) {
        console.error('Error processing gaze data:', error);
        setErrorMessage(ERROR_MESSAGES.PREDICTION_FAILED);
        setShowErrorDialog(true);
      }
    } else {
      // 手動評価モードの場合
      if (isEyeTrackerConnected) {
        dataManager.stopCollection();
        const gazeBuffer = dataManager.clearGazeBuffer();
        sessionManager.setCurrentGazeData(gazeBuffer);
      }
      setPhase(PHASES.CHOOSING);  // 手動選択フェーズへ
    }
  }, [phase, interactionMode, isEyeTrackerConnected, images, prompt, negativePrompt, currentStep, fixedPlacement]);

  const handleEnterPress = useCallback(async () => {
    if (phase !== PHASES.SELECTION || interactionMode !== 'gaze') return;
  
    setPhase(PHASES.PROCESSING);
    dataManager.stopCollection();
    const gazeBuffer = dataManager.clearGazeBuffer();
    
    if (!gazeBuffer || gazeBuffer.length < 10) {
      setErrorMessage('Not enough gaze data collected');
      setShowErrorDialog(true);
      return;
    }
  
    try {
      const endTime = new Date().toISOString();
      const durationSeconds = calculateDurationInSeconds(sessionStartTime, endTime);
  
      const result = await predictionService.getPrediction(gazeBuffer);
      const finalImageIndex = result.predictedIndex;
      const finalImage = images[finalImageIndex];
  
      // 最終データを保存
      const finalData = {
        images: images,
        prompt,
        generation: currentStep,
        selectedImage: finalImageIndex,
        prediction: result.predictedIndex,
        predictionConfidence: result.confidence,
        timestamp: new Date().toISOString()
      };
  
      // 最終ステップデータを保存
      await dataManager.saveStepData(
        sessionManager.sessionId,
        currentStep,
        gazeBuffer,
        finalData
      );
  
      // セッションメタデータを更新
      const finalMetadata = {
        evaluationMethod: interactionMode,
        modelId: selectedModel,
        prompt: prompt,
        negativePrompt: negativePrompt || null,
        fixedPlacement: fixedPlacement,
        eyeTrackerConnected: isEyeTrackerConnected,
        timestamp: endTime,
        startTime: formatJapanTime(sessionStartTime),
        endTime: formatJapanTime(endTime),
        sessionDurationSeconds: durationSeconds,  // 秒単位のduration
        finalSelectedImage: finalImage ? {
          ...finalImage,
          index: finalImageIndex,
          prediction: result.predictedIndex,
          confidence: result.confidence
        } : null
      };
  
      console.log('Saving final session metadata:', finalMetadata);
  
      await dataManager.saveSessionMetadata(sessionManager.sessionId, finalMetadata);
  
      // 最終選択画像を設定して結果画面へ
      setFinalSelectedImage(finalImage);
      setPrediction(result.predictedIndex);
      setPredictionConfidence(result.confidence);
      setPhase(PHASES.FINAL_RESULT);
    } catch (error) {
      console.error('Error in final prediction:', error);
      setErrorMessage(ERROR_MESSAGES.PREDICTION_FAILED);
      setShowErrorDialog(true);
    }
  }, [phase, interactionMode, images, prompt, currentStep, selectedModel, 
      negativePrompt, fixedPlacement, isEyeTrackerConnected, sessionStartTime]);

  // 画像選択のハンドラー
  const handleImageSelect = useCallback(async (imageIndex) => {
    if (phase === PHASES.CHOOSING) {
      setSelectedImage(imageIndex);
      setPhase(PHASES.CONFIRMING);
    }
  }, [phase]);

  // 進化を継続するハンドラー
  const handleContinueEvolution = useCallback(async () => {
    setPhase(PHASES.PROCESSING);
    
    try {
      const stepData = {
        images: images,
        prompt,
        generation: currentStep,
        timestamp: new Date().toISOString(),
        selectedImage,
        prediction: prediction,
        predictionConfidence: predictionConfidence
      };
  
      // 現在のステップのデータを保存
      try {
        const gazeBuffer = sessionManager.getCurrentGazeData();
        await dataManager.saveStepData(
          sessionManager.sessionId,
          currentStep,
          gazeBuffer,
          stepData
        );
      } catch (saveError) {
        console.error('Error saving data:', saveError);
      }
  
      // 選択された画像を保持
      const selectedImageData = images[selectedImage];
  
      // 4枚の画像を生成（選択された画像を含む）
      const result = await imageGenerationService.generateImages(
        prompt,
        negativePrompt,
        selectedImageData.latent_vector,
        currentStep + 1,  // generation
        4  // numImages
      );
  
      if (result && result.images) {
        // 生成された画像をマップ
        const newImages = result.images.map(img => ({
          ...img,
          src: imageGenerationService.getImageUrl(
            sessionManager.sessionId,
            currentStep + 1,
            img.url.split('/').pop()
          )
        }));
  
        // 配置設定に基づいて画像を並べ替え
        let displayImages;
        if (fixedPlacement) {
          // 選択された画像（newImages[0]）を左上に固定
          displayImages = newImages;
        } else {
          // 選択された画像をランダムな位置に配置
          const selectedImage = newImages[0];
          const otherImages = newImages.slice(1);
          const insertIndex = Math.floor(Math.random() * 4);
          displayImages = [...otherImages];
          displayImages.splice(insertIndex, 0, selectedImage);
        }
  
        setImages(displayImages);
        setCurrentStep(prev => prev + 1);
        setPhase(PHASES.SELECTION);
        
        // 視線データの収集を再開
        if (isEyeTrackerConnected) {
          dataManager.startCollection();
        }
      }
    } catch (error) {
      console.error('Error in evolution process:', error);
      setErrorMessage('Failed to generate next generation');
      setShowErrorDialog(true);
      setPhase(PHASES.SELECTION);
    }
  }, [selectedImage, images, prompt, negativePrompt, currentStep, fixedPlacement, isEyeTrackerConnected]);

  // 選択して終了するハンドラー
  const handleFinishWithSelection = useCallback(async () => {
    try {
      const endTime = new Date().toISOString();
      const durationSeconds = calculateDurationInSeconds(sessionStartTime, endTime);

      const finalData = {
        images: images,
        prompt,
        generation: currentStep,
        selectedImage: selectedImage,
        prediction: prediction,
        predictionConfidence: predictionConfidence,
        timestamp: endTime
      };

      await dataManager.saveStepData(
        sessionManager.sessionId,
        currentStep,
        sessionManager.getCurrentGazeData(),
        finalData
      );

      // 最終選択画像を設定
      const finalImage = images[selectedImage];
      // セッションメタデータを更新（所要時間を含む）
      const finalMetadata = {
        evaluationMethod: interactionMode,
        modelId: selectedModel,
        prompt,
        negativePrompt: negativePrompt || null,
        fixedPlacement: fixedPlacement,
        eyeTrackerConnected: isEyeTrackerConnected,
        timestamp: endTime,
        startTime: formatJapanTime(sessionStartTime),
        endTime: formatJapanTime(endTime),
        sessionDurationSeconds: durationSeconds,
        finalSelectedImage: {
          ...finalImage,
          index: selectedImage,
          prediction: prediction,
          confidence: predictionConfidence
        }
      };

      console.log('Saving final session metadata (manual):', finalMetadata);

      await dataManager.saveSessionMetadata(
        sessionManager.sessionId, 
        finalMetadata
      );

      setFinalSelectedImage(finalImage);
      setPhase(PHASES.FINAL_RESULT);
    } catch (error) {
      console.error('Error saving final selection:', error);
      setErrorMessage('Failed to save final selection');
      setShowErrorDialog(true);
    }
}, [selectedImage, prediction, predictionConfidence, currentStep, images, prompt, 
    interactionMode, selectedModel, negativePrompt, fixedPlacement, isEyeTrackerConnected, 
    sessionStartTime]);

  const handleExitToStart = useCallback(() => {
    setSessionStarted(false);
    setPhase(PHASES.SELECTION);
    setSelectedImage(null);
    setPrediction(null);
    setPredictionConfidence(null);
    setCurrentStep(0);
    setPrompt('');
    setNegativePrompt('');
    setInteractionMode(null);
    setFinalSelectedImage(null);
    dataManager.stopCollection();
  }, []);
  
  // セッション終了
  const handleEndSession = useCallback(() => {
    setSessionStarted(false);
    setPhase(PHASES.SELECTION);
    setSelectedImage(null);
    setPrediction(null);
    setPredictionConfidence(null);
    setCurrentStep(0);
    setPrompt('');
    setNegativePrompt('');
    setInteractionMode(null);
    dataManager.stopCollection();
  }, []);

  return (
    <Container maxWidth={false} sx={{ height: '100vh', py: 2 }}>
      {!sessionStarted ? (
        <StartScreen
          models={AVAILABLE_MODELS}
          selectedModel={selectedModel}
          onModelSelect={setSelectedModel}
          onStartSession={handleStartSession}
          isConnected={isEyeTrackerConnected}
        />
      ) : (
        <>
          {phase !== PHASES.FINAL_RESULT && (
            <>
              <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h5">
                  Eye Preference Evolution System
                </Typography>
                <Box>
                  <Typography color={isEyeTrackerConnected ? 'success.main' : 'error.main'}>
                    Eye Tracker: {isEyeTrackerConnected ? 'Connected' : 'Disconnected'}
                  </Typography>
                  {isDebugMode && <Typography>Debug Mode Active (Ctrl+D to toggle)</Typography>}
                </Box>
              </Box>
  
              <Box sx={{ mb: 2 }}>
                <Typography variant="body1">
                  Current Prompt: {prompt}
                </Typography>
                {negativePrompt && (
                  <Typography variant="body2" color="text.secondary">
                    Negative Prompt: {negativePrompt}
                  </Typography>
                )}
              </Box>
  
              <LinearProgress 
                variant="determinate" 
                value={currentStep * 10} 
                sx={{ mb: 2 }} 
              />
            </>
          )}
  
          {phase === PHASES.SELECTION && (
            <ImageGallery 
              images={images}
              gazeData={isDebugMode ? gazeData : null}
            />
          )}
  
          {phase === PHASES.CHOOSING && interactionMode === 'explicit' && (
            <SelectionDisplay 
              images={images}
              onImageSelect={handleImageSelect}
            />
          )}
  
          {phase === PHASES.CONFIRMING && interactionMode === 'explicit' && (
            <SelectionConfirmation
              selectedImage={selectedImage}
              predictedIndex={null}
              confidence={null}
              onNext={handleContinueEvolution}
              onFinish={handleFinishWithSelection}
            />
          )}
  
          {phase === PHASES.FINAL_RESULT && finalSelectedImage && (
            <FinalResultDisplay 
              selectedImage={finalSelectedImage}
              prompt={prompt}
              predictionConfidence={predictionConfidence}
              isPredicted={interactionMode === 'gaze'}
              onExit={handleExitToStart}
            />
          )}
  
          {phase !== PHASES.FINAL_RESULT && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'between' }}>
              <Typography>
                Generation: {currentStep}
              </Typography>
            </Box>
          )}
        </>
      )}
  
      {/* エラーダイアログ */}
      {showErrorDialog && (
        <Box sx={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          bgcolor: 'background.paper',
          boxShadow: 24,
          p: 4,
          borderRadius: 1,
          zIndex: 1000,
          minWidth: 300,
        }}>
          <Typography variant="h6" component="h2" gutterBottom>
            Error
          </Typography>
          <Typography sx={{ mt: 2 }}>
            {errorMessage}
          </Typography>
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button onClick={() => setShowErrorDialog(false)} color="primary">
              OK
            </Button>
          </Box>
        </Box>
      )}
    </Container>
  );
}

export default App;
