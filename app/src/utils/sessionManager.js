class SessionManager {
  constructor() {
      this.sessionId = null;
      this.stepCount = 0;
      this.startTime = null;
      this.currentGazeData = null;  // 現在の視線データを保持
  }

  startNewSession() {
      this.sessionId = this.generateSessionId();
      this.stepCount = 0;
      this.startTime = new Date();
      this.currentGazeData = null;
      return this.sessionId;
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
