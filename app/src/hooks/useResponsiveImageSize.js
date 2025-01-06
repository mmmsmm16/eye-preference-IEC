// useResponsiveImageSize.js
import { useState, useEffect, useCallback } from 'react';

export const useResponsiveImageSize = (mode = 'selection') => {
  const [size, setSize] = useState({
    imageHeight: 512,
    imageWidth: 512
  });

  const updateSize = useCallback(() => {
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    
    const margins = {
      selection: {
        vertical: 200,  // ヘッダーとフッターのスペースを確保
        horizontal: 64
      },
      result: {
        vertical: 240,  // ボタン用の追加スペースを考慮
        horizontal: 64
      }
    };

    const spacing = mode === 'selection' ? 32 : 24;
    const currentMargins = margins[mode];
    
    // 利用可能な空間を計算
    const availableHeight = vh - currentMargins.vertical;
    const availableWidth = vw - currentMargins.horizontal;
    
    // 2x2グリッドの1つの画像の最大サイズを計算
    const maxHeight = Math.floor((availableHeight - spacing) / 2);
    const maxWidth = Math.floor((availableWidth - spacing) / 2);
    
    // 正方形を維持するために小さい方のサイズに合わせる
    const size = Math.min(maxHeight, maxWidth, 512);
    
    setSize({
      imageHeight: size,
      imageWidth: size
    });
  }, [mode]);

  useEffect(() => {
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [updateSize]);

  return size;
};
