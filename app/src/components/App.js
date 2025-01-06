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

// アプリケーションのフェーズを定義
const PHASES = {
  SELECTION: 'selection',    // 視線/明示的選択フェーズ
  CHOOSING: 'choosing',      // 手動選択フェーズ
  PROCESSING: 'processing',  // 予測/生成処理フェーズ
  CONFIRMING: 'confirming', // 新しい確認フェーズ - 選択後の決定用
  RESULT: 'result'          // 結果表示フェーズ
};

const SPACE_KEY = 32;
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

  // セッション開始の処理
  const handleStartSession = useCallback(async ({
    interactionMode: mode,
    selectedModel: model,
    prompt: inputPrompt,
    negativePrompt: inputNegativePrompt,
    fixedPlacement: placement,
  }) => {
    try {
      setInteractionMode(mode);
      setSelectedModel(model);
      setPrompt(inputPrompt);
      setNegativePrompt(inputNegativePrompt);
      setFixedPlacement(placement);
      
      const sessionId = sessionManager.startNewSession();
      setSessionStarted(true);
      setCurrentStep(0);
      setPreviousSelectedImage(null);
      
      // アイトラッカーが接続されている場合は視線データ収集を開始
      if (isEyeTrackerConnected) {
        dataManager.startCollection();
      }
      
      // 初期画像の生成
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
      if (event.keyCode === SPACE_KEY && phase === PHASES.SELECTION) {
        handleSpacePress();
      } else if (event.keyCode === DEBUG_KEY && event.ctrlKey) {
        setIsDebugMode(prev => !prev);
      }
    };
  
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [phase]);

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
        const result = await predictionService.getPrediction(gazeBuffer);
        setPrediction(result.predictedIndex);
        setPredictionConfidence(result.confidence);
        setSelectedImage(result.predictedIndex);
        setPhase(PHASES.CONFIRMING);
      } catch (error) {
        console.error('Prediction error:', error);
        setErrorMessage(ERROR_MESSAGES.PREDICTION_FAILED);
        setShowErrorDialog(true);
      }
    } else {
      // 手動評価モードの場合
      if (isEyeTrackerConnected) {
        // 視線データを保存してからCHOOSINGフェーズへ
        dataManager.stopCollection();
        const gazeBuffer = dataManager.clearGazeBuffer();
        sessionManager.setCurrentGazeData(gazeBuffer);
      }
      setPhase(PHASES.CHOOSING);
    }
  }, [phase, interactionMode, isEyeTrackerConnected]);

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
  
      // 3枚の新しい画像を生成
      const result = await imageGenerationService.generateImages(
        prompt,
        negativePrompt,
        selectedImageData.latent_vector,
        currentStep + 1,
        3  // 生成する画像の数を3に変更
      );
  
      if (result && result.images) {
        // 生成された新しい画像を処理
        const newImages = result.images.map(img => ({
          ...img,
          src: imageGenerationService.getImageUrl(
            sessionManager.sessionId,
            currentStep + 1,
            img.url.split('/').pop()
          )
        }));
  
        // 選択された画像を含む配列を作成
        let combinedImages;
        if (fixedPlacement) {
          // 固定配置: 選択された画像を左上に配置
          combinedImages = [selectedImageData, ...newImages];
        } else {
          // ランダム配置: 選択された画像をランダムな位置に配置
          const insertIndex = Math.floor(Math.random() * 4);
          combinedImages = [...newImages];
          combinedImages.splice(insertIndex, 0, selectedImageData);
        }
  
        setImages(combinedImages);
        setPreviousSelectedImage(selectedImageData);
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
      // 最終選択のデータを準備
      const finalData = {
        images: images,
        prompt,
        generation: currentStep,
        selectedImage: selectedImage,
        prediction: prediction,
        predictionConfidence: predictionConfidence,
        timestamp: new Date().toISOString()
      };
  
      // 最終データを保存
      await dataManager.saveStepData(
        sessionManager.sessionId,
        currentStep,
        sessionManager.getCurrentGazeData(),
        finalData
      );
  
      // セッション終了
      handleEndSession();
    } catch (error) {
      console.error('Error saving final selection:', error);
      setErrorMessage('Failed to save final selection');
      setShowErrorDialog(true);
    }
  }, [selectedImage, prediction, predictionConfidence, currentStep, images, prompt]);

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

          {/* 選択フェーズ - 視線データ収集中 */}
          {phase === PHASES.SELECTION && (
            <ImageGallery 
              images={images}
              gazeData={isDebugMode ? gazeData : null}
            />
          )}

          {/* 手動選択フェーズ */}
          {phase === PHASES.CHOOSING && interactionMode === 'explicit' && (
            <SelectionDisplay 
              images={images}
              onImageSelect={handleImageSelect}
            />
          )}

          {/* 確認フェーズ - 新しく追加 */}
          {phase === PHASES.CONFIRMING && (
            <SelectionConfirmation
              selectedImage={selectedImage}
              predictedIndex={interactionMode === 'gaze' ? prediction : null}
              confidence={interactionMode === 'gaze' ? predictionConfidence : null}
              onNext={handleContinueEvolution}
              onFinish={handleFinishWithSelection}
            />
          )}

          {/* 進行状況表示 */}
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'between' }}>
            <Typography>
              Generation: {currentStep}
            </Typography>
          </Box>
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
