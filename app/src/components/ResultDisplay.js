import React from 'react';
import { Box, Grid, Card, CardMedia, Typography, Button } from '@mui/material';
import { useResponsiveImageSize } from '../hooks/useResponsiveImageSize';

const ResultDisplay = ({ 
  images, 
  predictedIndex, 
  selectedIndex,
  confidence,
  onNext,
  onEnd
}) => {
  const { imageHeight, imageWidth } = useResponsiveImageSize('result');

  return (
    <Box sx={{ 
      minHeight: 'calc(100vh - 200px)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center'
    }}>
      <Typography variant="h6" align="center" sx={{ mb: 4 }}>
        Results Comparison
      </Typography>
      
      <Box sx={{ 
        maxWidth: imageWidth * 2 + 24,
        width: '100%',
        margin: '0 auto'
      }}>
        <Grid container spacing={3}>
          {images.map((image, index) => (
            <Grid item xs={6} key={index}>
              <Card sx={{ 
                width: imageWidth,
                height: imageWidth,
                border: (() => {
                  if (index === selectedIndex && index === predictedIndex) {
                    return '4px solid #4CAF50';  // 緑: 予測と選択が一致
                  } else if (index === selectedIndex) {
                    return '4px solid #2196F3';  // 青: ユーザーの選択
                  } else if (index === predictedIndex) {
                    return '4px solid #FFA726';  // オレンジ: システムの予測
                  }
                  return 'none';
                })()
              }}>
                <Box sx={{ 
                  width: '100%',
                  height: '100%',
                  position: 'relative'
                }}>
                  <CardMedia
                    component="img"
                    image={image.src}
                    alt={`Image ${index + 1}`}
                    sx={{ 
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover'
                    }}
                  />
                  {(index === selectedIndex || index === predictedIndex) && (
                    <Box
                      sx={{
                        position: 'absolute',
                        top: 10,
                        right: 10,
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 1,
                        zIndex: 1
                      }}
                    >
                      {index === selectedIndex && (
                        <Box
                          sx={{
                            backgroundColor: '#2196F3',
                            color: 'white',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            boxShadow: 2
                          }}
                        >
                          Your Choice
                        </Box>
                      )}
                      {index === predictedIndex && (
                        <Box
                          sx={{
                            backgroundColor: '#FFA726',
                            color: 'white',
                            padding: '4px 8px',
                            borderRadius: '4px',
                            fontWeight: 'bold',
                            boxShadow: 2
                          }}
                        >
                          Predicted Choice 
                          {confidence && ` (${(confidence * 100).toFixed(1)}%)`}
                        </Box>
                      )}
                    </Box>
                  )}
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>

      {selectedIndex === predictedIndex && (
        <Typography 
          variant="h6" 
          align="center" 
          sx={{ mt: 3, color: '#4CAF50' }}
        >
          Prediction matches your choice! 🎯
        </Typography>
      )}

      <Box sx={{ 
        mt: 4,
        display: 'flex', 
        justifyContent: 'center',
        gap: 3
      }}>
        <Button variant="contained" color="primary" onClick={onNext}>
          Next Trial
        </Button>
        <Button variant="outlined" color="secondary" onClick={onEnd}>
          End Session
        </Button>
      </Box>
    </Box>
  );
};

export default ResultDisplay;
