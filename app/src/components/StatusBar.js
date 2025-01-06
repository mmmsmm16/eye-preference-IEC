import React from 'react';
import { Box, Paper, Typography, Chip } from '@mui/material';
import { Wifi, WifiOff, Error as ErrorIcon } from '@mui/icons-material';

const StatusBar = ({ isEyeTrackerConnected, isProcessing, error }) => {
  return (
    <Paper 
      elevation={3} 
      sx={{ 
        p: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}
    >
      {/* 接続状態 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Chip
          icon={isEyeTrackerConnected ? <Wifi /> : <WifiOff />}
          label={`Eye Tracker: ${isEyeTrackerConnected ? 'Connected' : 'Disconnected'}`}
          color={isEyeTrackerConnected ? 'success' : 'error'}
          variant="outlined"
        />
        
        <Chip
          label={isProcessing ? 'Processing...' : 'Ready'}
          color={isProcessing ? 'primary' : 'success'}
          variant="outlined"
        />
      </Box>

      {/* エラー表示 */}
      {error && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <ErrorIcon color="error" />
          <Typography color="error" variant="body2">
            {error}
          </Typography>
        </Box>
      )}
    </Paper>
  );
};

export default StatusBar;
