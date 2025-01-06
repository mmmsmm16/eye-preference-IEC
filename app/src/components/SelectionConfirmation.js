import React from 'react';
import {
  Box,
  Button,
  Typography,
  Paper,
  Grid
} from '@mui/material';
import { NavigateNext, CheckCircle } from '@mui/icons-material';

const SelectionConfirmation = ({ 
  selectedImage,
  onNext,
  onFinish,
  predictedIndex = null,
  confidence = null
}) => {
  return (
    <Paper 
      sx={{ 
        p: 3, 
        mt: 3,
        maxWidth: '600px',
        mx: 'auto'
      }}
    >
      <Grid container direction="column" spacing={3}>
        <Grid item>
          <Typography variant="h6" align="center" gutterBottom>
            {predictedIndex !== null ? 'Prediction Complete' : 'Image Selected'}
          </Typography>
          
          <Typography align="center" color="text.secondary">
            {predictedIndex !== null 
              ? `Predicted choice: Image ${predictedIndex + 1}${confidence ? ` (${(confidence * 100).toFixed(1)}% confidence)` : ''}`
              : `You selected Image ${selectedImage + 1}`
            }
          </Typography>
        </Grid>

        <Grid item>
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'center',
            gap: 3
          }}>
            <Button
              variant="contained"
              color="primary"
              endIcon={<NavigateNext />}
              onClick={onNext}
              size="large"
            >
              Continue Evolution
            </Button>
            <Button
              variant="contained"
              color="success"
              endIcon={<CheckCircle />}
              onClick={onFinish}
              size="large"
            >
              Select and Finish
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
};

export default SelectionConfirmation;
