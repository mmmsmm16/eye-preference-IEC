import React from 'react';
import { Box, Grid, Card, CardMedia, Typography } from '@mui/material';
import { useResponsiveImageSize } from '../hooks/useResponsiveImageSize';

const SelectionDisplay = ({ images, onImageSelect }) => {
  const { imageHeight, imageWidth } = useResponsiveImageSize('selection');

  return (
    <Box sx={{ 
      minHeight: 'calc(100vh - 160px)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center'  // 垂直方向の中央揃え
    }}>
      <Typography variant="h6" align="center" sx={{ mb: 4 }}>
        Please select the image you prefer
      </Typography>
      
      <Box sx={{ 
        maxWidth: imageWidth * 2 + 32,
        width: '100%',
        margin: '0 auto'
      }}>
        <Grid container spacing={4}>
        {images.map((image, index) => (
          <Grid item xs={6} key={index}>
            <Card 
              sx={{ 
                width: imageWidth,
                height: imageWidth,
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'scale(1.02)',
                }
              }}
              onClick={() => onImageSelect(index)}
            >
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
                <Box
                  sx={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    bgcolor: 'rgba(0, 0, 0, 0.7)',
                    color: 'white',
                    p: 1,
                    textAlign: 'center'
                  }}
                >
                  Click to select
                </Box>
              </Box>
            </Card>
          </Grid>
        ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default SelectionDisplay;
