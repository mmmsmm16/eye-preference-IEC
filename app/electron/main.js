const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs').promises;

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1600,
    height: 900,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, '../index.html'));
  mainWindow.webContents.openDevTools();

  // デバッグ用のログ出力
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Page failed to load:', errorCode, errorDescription);
  });

  mainWindow.webContents.on('did-finish-load', () => {
    console.log('Page loaded successfully');
  });
}

// IPCハンドラーの設定
function setupIpcHandlers() {
  ipcMain.handle('save-data', async (event, args) => {
    try {
      const { type, path: filePath, content } = args;
      
      // パスが存在することを確認
      const dir = path.dirname(filePath);
      await fs.mkdir(dir, { recursive: true });

      // データの保存
      await fs.writeFile(
        path.join(process.cwd(), filePath),
        content,
        'utf8'
      );

      return { success: true };
    } catch (error) {
      console.error('Error saving data:', error);
      throw error;
    }
  });

  // 画像セット読み込みハンドラー
  ipcMain.handle('load-image-sets', async () => {
    try {
      const imageDataPath = path.join(app.getAppPath(), 'image_data');
      const directories = await fs.readdir(imageDataPath);
      
      const imageSets = {};
      for (const dir of directories) {
        const dirPath = path.join(imageDataPath, dir);
        const stat = await fs.stat(dirPath);
        
        if (stat.isDirectory()) {
          imageSets[dir] = await loadImagesFromDirectory(dirPath);
        }
      }
      
      return imageSets;
    } catch (error) {
      console.error('Error loading image sets:', error);
      throw error;
    }
  });
}

// 画像セットを読み込むためのヘルパー関数
async function loadImagesFromDirectory(dirPath) {
  const images = [];
  const files = await fs.readdir(dirPath);
  
  for (const file of files) {
    if (file.match(/\.(jpg|jpeg|png|gif)$/i)) {
      const fullPath = path.join(dirPath, file);
      images.push({
        id: path.parse(file).name,
        path: fullPath
      });
    }
  }
  
  return images;
}


app.whenReady().then(() => {
  createWindow();
  setupIpcHandlers();  // IPCハンドラーを設定

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
