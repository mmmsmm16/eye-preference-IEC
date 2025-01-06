const { ipcRenderer } = window.require('electron');

class PredictionService {
  async getPrediction(gazeData) {
    try {
      // gazeDataを適切な形式に変換
      const formattedData = this.formatGazeData(gazeData);
      
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timestamp: formattedData.timestamp,
          left_x: formattedData.left_x,
          left_y: formattedData.left_y,
          right_x: formattedData.right_x,
          right_y: formattedData.right_y,
          model_name: 'lstm_method1',
          method: 'method1'
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`Prediction failed: ${errorData.detail || response.statusText}`);
      }

      const result = await response.json();
      console.log('Prediction result:', result);
      
      // predictions配列から最大値のインデックスを取得
      const predictedIndex = result.predictions.indexOf(Math.max(...result.predictions));
      
      return {
        predictedIndex,
        confidence: result.confidence,
        predictions: result.predictions
      };

    } catch (error) {
      console.error('Error during prediction:', error);
      throw error;
    }
  }

  formatGazeData(gazeData) {
    // gazeDataを必要な形式に変換
    const formatted = {
      timestamp: [],
      left_x: [],
      left_y: [],
      right_x: [],
      right_y: []
    };

    gazeData.forEach(data => {
      formatted.timestamp.push(data.timestamp);
      formatted.left_x.push(data.left_x);
      formatted.left_y.push(data.left_y);
      formatted.right_x.push(data.right_x);
      formatted.right_y.push(data.right_y);
    });

    return formatted;
  }
}

export default new PredictionService();
