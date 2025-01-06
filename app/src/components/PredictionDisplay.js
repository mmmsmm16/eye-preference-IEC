import React from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  LinearProgress, 
  CircularProgress 
} from '@mui/material';

const PredictionDisplay = ({ predictions, confidence, isProcessing }) => {
  const positions = ['Top Left', 'Top Right', 'Bottom Left', 'Bottom Right'];

  if (isProcessing) {
    return (
      <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          gap: 2
        }}>
          <CircularProgress />
          <Typography>Processing gaze data...</Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Prediction Results
      </Typography>

      {predictions ? (
        <>
          {/* 予測結果の表示 */}
          <Box sx={{ mt: 2 }}>
            {predictions.map((prob, index) => (
              <Box key={index} sx={{ mb: 2 }}>
                <Box sx={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  mb: 1
                }}>
                  <Typography variant="body2">
                    {positions[index]}
                  </Typography>
                  <Typography variant="body2">
                    {(prob * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress 
                  variant="determinate" 
                  value={prob * 100}
                  sx={{
                    height: 8,
                    borderRadius: 1,
                    backgroundColor: 'action.hover',
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 1,
                      backgroundColor: index === predictions.indexOf(Math.max(...predictions)) 
                        ? 'primary.main' 
                        : 'secondary.main'
                    }
                  }}
                />
              </Box>
            ))}
          </Box>

          {/* 信頼度の表示 */}
          {confidence !== null && (
            <Box sx={{ mt: 3, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Typography variant="body2" color="textSecondary">
                Confidence: {(confidence * 100).toFixed(1)}%
              </Typography>
            </Box>
          )}
        </>
      ) : (
        <Typography color="textSecondary">
          Waiting for gaze data...
        </Typography>
      )}
    </Paper>
  );
};

export default PredictionDisplay;
