import sessionManager from './sessionManager';

class DataManager {
  constructor() {
    this.gazeDataBuffer = [];
    this.isCollecting = false;
    this.baseUrl = 'http://host.docker.internal:8000';
    this.retryConfig = {
      maxRetries: 3,
      retryDelay: 1000
    };
  }

  async retryOperation(operation) {
    let lastError;
    
    for (let i = 0; i < this.retryConfig.maxRetries; i++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        if (i < this.retryConfig.maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, this.retryConfig.retryDelay * (i + 1)));
          continue;
        }
      }
    }
    
    throw lastError;
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

  getEvaluationMode(mode) {
    // 'explicit'を'manual'に変換
    return mode === 'explicit' ? 'manual' : mode;
  }

  async saveSessionMetadata(sessionId, metadata) {
    return this.retryOperation(async () => {
      try {
        const endpoint = `${this.baseUrl}/save-data/session-metadata`;
        // evaluationMethodを正規化
        const normalizedMetadata = {
          ...metadata,
          evaluationMethod: this.getEvaluationMode(metadata.evaluationMethod)
        };

        const requestPayload = {
          sessionId,
          metadata: normalizedMetadata
        };

        console.log('Saving session metadata:', {
          endpoint,
          sessionId,
          metadata: normalizedMetadata
        });

        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestPayload)
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          console.error('Server error response:', errorData);
          throw new Error(`Failed to save session metadata: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('Session metadata saved successfully:', result);
        return true;
      } catch (error) {
        console.error('Error saving session metadata:', error);
        throw error;
      }
    });
  }

  async saveStepData(sessionId, stepId, gazeData, trialData) {
    return this.retryOperation(async () => {
      try {
        console.log(`Saving data for session ${sessionId}, step ${stepId}`);
        
        // evaluationMethodを正規化
        const evaluationMethod = this.getEvaluationMode(sessionManager.interactionMode);
        
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
            selectedImageIndex: trialData.selectedImage,
            predictedIndex: trialData.prediction,
            predictionConfidence: trialData.predictionConfidence
          },
          evaluationMethod: evaluationMethod
        };

        const response = await fetch(`${this.baseUrl}/save-data/step-data`, {
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
        console.error('Error saving data:', error);
        throw error;
      }
    });
  }
}

export default new DataManager();
