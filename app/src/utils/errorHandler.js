class ErrorHandler {
    constructor() {
      this.errorBuffer = [];
      this.dataBackup = new Map();
    }
  
    async handleSaveError(sessionId, stepId, data, error) {
      console.error(`Save error for session ${sessionId}, step ${stepId}:`, error);
      
      // バックアップデータを保存
      this.dataBackup.set(`${sessionId}_${stepId}`, data);
  
      // LocalStorageにも保存を試みる
      try {
        localStorage.setItem(
          `backup_${sessionId}_${stepId}`, 
          JSON.stringify({
            timestamp: new Date().toISOString(),
            data: data
          })
        );
      } catch (e) {
        console.error('Failed to save backup:', e);
      }
  
      return {
        success: false,
        error: error.message,
        hasBackup: true,
        retryFunction: () => this.retrySave(sessionId, stepId)
      };
    }
  
    async retrySave(sessionId, stepId) {
      const data = this.dataBackup.get(`${sessionId}_${stepId}`);
      if (!data) {
        throw new Error('Backup data not found');
      }
  
      try {
        // 保存を再試行
        await dataManager.saveStepData(sessionId, stepId, data.gazeData, data.trialData);
        this.dataBackup.delete(`${sessionId}_${stepId}`);
        localStorage.removeItem(`backup_${sessionId}_${stepId}`);
        return true;
      } catch (error) {
        console.error('Retry save failed:', error);
        return false;
      }
    }
  
    // セッション終了時のバックアップ確認
    async checkPendingBackups() {
      const backups = Array.from(this.dataBackup.entries());
      if (backups.length > 0) {
        return {
          hasPendingBackups: true,
          count: backups.length,
          retryAll: async () => {
            const results = await Promise.all(
              backups.map(([key, data]) => {
                const [sessionId, stepId] = key.split('_');
                return this.retrySave(sessionId, stepId);
              })
            );
            return results.every(result => result);
          }
        };
      }
      return { hasPendingBackups: false };
    }
  }
  
  export default new ErrorHandler();
