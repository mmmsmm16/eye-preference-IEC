import React from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardMedia
} from '@mui/material';
import { ExitToApp } from '@mui/icons-material';

const FinalResultDisplay = ({ 
  selectedImage,
  prompt,
  predictionConfidence = null,
  isPredicted = false,
  onExit
}) => {
  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center',
      gap: 4,
      padding: 4
    }}>
      <Typography variant="h4" gutterBottom>
        Final Result
      </Typography>

      <Typography variant="h6" color="text.secondary" gutterBottom>
        {isPredicted ? "Predicted Preferred Image" : "Selected Image"}
      </Typography>

      <Box sx={{ maxWidth: 512, width: '100%' }}>
        <Card sx={{ width: '100%', aspectRatio: '1/1' }}>
          <CardMedia
            component="img"
            image={selectedImage.src}
            alt="Final selected image"
            sx={{ 
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
        </Card>
      </Box>

      <Box sx={{ textAlign: 'center', mt: 2 }}>
        <Typography variant="body1" gutterBottom>
          Prompt: {prompt}
        </Typography>
        {isPredicted && predictionConfidence !== null && (
          <Typography variant="body2" color="text.secondary">
            Prediction Confidence: {(predictionConfidence * 100).toFixed(1)}%
          </Typography>
        )}
      </Box>

      <Button
        variant="contained"
        color="primary"
        startIcon={<ExitToApp />}
        onClick={onExit}
        size="large"
        sx={{ mt: 4 }}
      >
        Exit to Start Screen
      </Button>
    </Box>
  );
};

export default FinalResultDisplay;
