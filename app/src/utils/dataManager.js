class DataManager {
  constructor() {
    this.gazeDataBuffer = [];
    this.isCollecting = false;
    this.baseUrl = 'http://host.docker.internal:8000';
  }

  startCollection() {
    this.isCollecting = true;
    this.gazeDataBuffer = [];
    console.log('Started gaze data collection');
  }

  stopCollection() {
    this.isCollecting = false;
    console.log(`Stopped gaze data collection. Collected ${this.gazeDataBuffer.length} samples`);
  }

  addGazeData(data) {
    if (!this.isCollecting) return;
    
    this.gazeDataBuffer.push({
      timestamp: data.timestamp,
      left_x: data.left_x,
      left_y: data.left_y,
      right_x: data.right_x,
      right_y: data.right_y
    });
  }

  clearGazeBuffer() {
    const buffer = [...this.gazeDataBuffer];
    this.gazeDataBuffer = [];
    console.log(`Cleared buffer, returning ${buffer.length} samples`);
    return buffer;
  }

  async saveStepData(sessionId, stepId, gazeData, trialData) {
    try {
      const endpoint = `${this.baseUrl}/save-data/step-data`;
      console.log('Attempting to save data to:', endpoint);
      
      // 選択情報を含めたデータの準備
      const saveData = {
        sessionId,
        stepId,
        timestamp: new Date().toISOString(),
        gazeData: gazeData ? {
          sampleCount: gazeData.length,
          duration: gazeData.length > 0 ? 
            gazeData[gazeData.length - 1].timestamp - gazeData[0].timestamp : 0,
          data: gazeData
        } : null,
        images: trialData.images,
        prompt: trialData.prompt,
        generation: trialData.generation,
        selection: {
          selectedImageIndex: trialData.selectedImage,         // 選択された画像のインデックス
          predictedIndex: trialData.prediction,                // 予測されたインデックス（視線評価の場合）
          predictionConfidence: trialData.predictionConfidence // 予測の信頼度（視線評価の場合）
        }
      };

      console.log('Request payload:', saveData);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(saveData)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Server error response:', errorData);
        throw new Error(`Failed to save data: ${response.statusText}`);
      }

      const result = await response.json();
      console.log('Data saved successfully:', result);
      return true;

    } catch (error) {
      console.error('Error in saveStepData:', error);
      throw error;
    }
  }
}

export default new DataManager();
