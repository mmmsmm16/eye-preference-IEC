class SessionManager {
    constructor() {
      this.sessionId = null;
      this.stepCount = 0;
      this.startTime = null;
      this.currentGazeData = null;
      this.interactionMode = null;
      this.baseUrl = 'http://host.docker.internal:8000';
    }

    getEvaluationMode() {
        // 'explicit'を'manual'に変換
        return this.interactionMode === 'explicit' ? 'manual' : this.interactionMode;
    }

    async startNewSession() {
        try {
            const currentNumber = await this.getNextSessionNumber();
            
            // セッションIDを生成（ディレクトリパスは含まない）
            this.sessionId = currentNumber.toString();
            
            this.stepCount = 0;
            this.startTime = new Date();
            this.currentGazeData = null;
            
            return this.sessionId;
        } catch (error) {
            console.error('Error creating new session:', error);
            throw error;
        }
    }

    async getNextSessionNumber() {
        try {
          const mode = this.getEvaluationMode();
          const response = await fetch(`${this.baseUrl}/save-data/get-next-session-number/${mode}`);
          
          if (!response.ok) {
            throw new Error('Failed to get session number');
          }
          
          const data = await response.json();
          return data.nextNumber;
        } catch (error) {
          console.error('Error getting next session number:', error);
          throw error;
        }
    }

    setInteractionMode(mode) {
        this.interactionMode = mode;
        console.log('Set interaction mode:', mode); // デバッグログを追加
    }

    setCurrentGazeData(gazeData) {
        this.currentGazeData = gazeData;
    }

    getCurrentGazeData() {
        return this.currentGazeData;
    }

    clearCurrentGazeData() {
        const data = this.currentGazeData;
        this.currentGazeData = null;
        return data;
    }

    generateSessionId() {
        const date = new Date();
        return `session_${date.getFullYear()}${(date.getMonth() + 1).toString().padStart(2, '0')}${date.getDate().toString().padStart(2, '0')}_${date.getHours().toString().padStart(2, '0')}${date.getMinutes().toString().padStart(2, '0')}${date.getSeconds().toString().padStart(2, '0')}`;
    }

    incrementStep() {
        this.stepCount++;
        return this.stepCount;
    }

    getCurrentSessionInfo() {
        return {
            sessionId: this.sessionId,
            currentStep: this.stepCount,
            startTime: this.startTime,
            duration: this.startTime ? new Date() - this.startTime : 0
        };
    }
}

export default new SessionManager();
