import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Grid,
  TextField,
  CircularProgress,
  ToggleButtonGroup,
  ToggleButton,
  FormControlLabel,
  Switch
} from '@mui/material';
import { VisibilityOutlined, TouchAppOutlined } from '@mui/icons-material';

const StartScreen = ({
  onStartSession,
  isConnected,
  models,
  selectedModel,
  onModelSelect,
}) => {
  const [interactionMode, setInteractionMode] = useState('explicit');
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [fixedPlacement, setFixedPlacement] = useState(true); // 選択画像の固定配置設定

  const handleInteractionModeChange = (event, newMode) => {
    if (newMode !== null) {
      setInteractionMode(newMode);
      if (newMode === 'explicit') {
        onModelSelect('');
      }
    }
  };

  const handleStartClick = async () => {
    if (!prompt.trim()) return;

    setIsLoading(true);
    try {
      await onStartSession({
        interactionMode,
        selectedModel,
        prompt: prompt.trim(),
        negativePrompt: negativePrompt.trim() || null,
        fixedPlacement, // 配置設定を追加
      });
    } catch (error) {
      console.error('Failed to start session:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, mt: 8 }}>
      <Typography variant="h4" gutterBottom>
        Eye Preference Evolution System
      </Typography>

      <Paper sx={{ p: 4, maxWidth: 600, width: '100%' }}>
        <Grid container spacing={3}>
          {/* 評価方式の選択 */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
              <Typography variant="subtitle1">
                Evaluation Method
              </Typography>
              <ToggleButtonGroup
                value={interactionMode}
                exclusive
                onChange={handleInteractionModeChange}
                aria-label="evaluation method"
                sx={{ width: 'fit-content' }}
              >
                <ToggleButton value="explicit" aria-label="manual evaluation" sx={{ px: 3, py: 1.5 }}>
                  <TouchAppOutlined sx={{ mr: 1 }} />
                  Manual Evaluation
                </ToggleButton>
                <ToggleButton value="gaze" aria-label="gaze-based evaluation" sx={{ px: 3, py: 1.5 }}>
                  <VisibilityOutlined sx={{ mr: 1 }} />
                  Gaze-based Evaluation
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          </Grid>

          {/* 配置設定 */}
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={fixedPlacement}
                    onChange={(e) => setFixedPlacement(e.target.checked)}
                    name="fixedPlacement"
                  />
                }
                label={fixedPlacement ? "Fixed placement (top-left)" : "Random placement"}
              />
            </Box>
          </Grid>

          {/* 既存のフィールド... */}
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Generation Prompt"
              multiline
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a description of the image you want to generate..."
              required
            />
          </Grid>

          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Negative Prompt (Optional)"
              multiline
              rows={2}
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="Enter elements you want to avoid in the generated images..."
            />
          </Grid>

          {/* モデル選択 - 視線ベースの評価の時のみ表示 */}
          {interactionMode === 'gaze' && (
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Model</InputLabel>
                <Select
                  value={selectedModel || ''}
                  onChange={(e) => onModelSelect(e.target.value)}
                  label="Model"
                >
                  {models.map((model) => (
                    <MenuItem key={model.id} value={model.id}>
                      {model.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          )}

          {/* 開始ボタン */}
          <Grid item xs={12}>
            <Button
              variant="contained"
              fullWidth
              onClick={handleStartClick}
              disabled={isLoading || !prompt.trim() || (interactionMode === 'gaze' && (!selectedModel || !isConnected))}
              sx={{ mt: 2 }}
            >
              {isLoading ? <CircularProgress size={24} color="inherit" /> : 'Start Evolution Session'}
            </Button>
          </Grid>

          {/* Eye Trackerの接続状態表示 */}
          <Grid item xs={12}>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              gap: 1,
              backgroundColor: 'background.paper',
              padding: 1,
              borderRadius: 1
            }}>
              <Box sx={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                backgroundColor: isConnected ? 'success.main' : 'error.main' 
              }} />
              {interactionMode === 'gaze' ? (
                <Typography 
                  color={isConnected ? 'success.main' : 'error.main'}
                  sx={{ textAlign: 'center' }}
                >
                  {isConnected ? 'Eye Tracker Connected' : 'Eye Tracker not connected. Required for gaze-based evaluation.'}
                </Typography>
              ) : (
                <Typography 
                  color={isConnected ? 'success.main' : 'text.secondary'}
                  sx={{ textAlign: 'center' }}
                >
                  {isConnected ? 'Eye Tracker Connected - Gaze data will be recorded' : 'Eye Tracker not connected - Optional for manual evaluation'}
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>
      </Paper>
    </Box>
  );
};

export default StartScreen;
