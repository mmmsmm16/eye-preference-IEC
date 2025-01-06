import React, { useRef, useEffect } from 'react';
import { Box, Grid, Card, CardMedia } from '@mui/material';
import { useResponsiveImageSize } from '../hooks/useResponsiveImageSize';

const ImageGallery = ({ images, gazeData }) => {
  const { imageHeight, imageWidth } = useResponsiveImageSize('selection');
  const containerRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!gazeData || !canvasRef.current || !containerRef.current) return;

    const canvas = canvasRef.current;
    const container = containerRef.current;
    const rect = container.getBoundingClientRect();

    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext('2d');

    // 視線位置を描画
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(
      gazeData.left_x * canvas.width,
      gazeData.left_y * canvas.height,
      5,
      0,
      2 * Math.PI
    );
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(
      gazeData.right_x * canvas.width,
      gazeData.right_y * canvas.height,
      5,
      0,
      2 * Math.PI
    );
    ctx.fillStyle = 'rgba(0, 0, 255, 0.5)';
    ctx.fill();
  }, [gazeData]);

  return (
    <Box 
      ref={containerRef} 
      sx={{ 
        minHeight: 'calc(100vh - 160px)',
        display: 'flex',
        alignItems: 'center', // 垂直方向の中央揃え
        justifyContent: 'center', // 水平方向の中央揃え
        position: 'relative'
      }}
    >
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          zIndex: 1
        }}
      />
      <Box sx={{ 
        maxWidth: imageWidth * 2 + 32,
        width: '100%'
      }}>
        <Grid 
          container 
          spacing={4}
        >
          {images.map((image, index) => (
            <Grid item xs={6} key={index}>
              <Card 
                sx={{ 
                  width: imageWidth,
                  height: imageWidth
                }}
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
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Box>
  );
};

export default ImageGallery;
