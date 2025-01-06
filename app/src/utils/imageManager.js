const { ipcRenderer } = window.require('electron');
const path = window.require('path');

class ImageManager {
  constructor() {
    this.imageSets = new Map();
    this.currentImageSet = null;
  }

  async loadImageSets() {
    try {
      // IPCを通じてimage_dataディレクトリの内容を取得
      const imageSets = await ipcRenderer.invoke('load-image-sets');
      console.log('Loaded image sets:', imageSets);
      this.imageSets = new Map(Object.entries(imageSets));
      return Array.from(this.imageSets.keys());
    } catch (error) {
      console.error('Error loading image sets:', error);
      return [];
    }
  }

  selectImageSet(imageSetId) {
    if (!this.imageSets.has(imageSetId)) {
      throw new Error(`Image set ${imageSetId} not found`);
    }
    this.currentImageSet = imageSetId;
    return this.imageSets.get(imageSetId);
  }

  getRandomImages(count = 4) {
    if (!this.currentImageSet) {
      throw new Error('No image set selected');
    }

    const images = this.imageSets.get(this.currentImageSet);
    if (!images || images.length < count) {
      throw new Error(`Not enough images in set ${this.currentImageSet}`);
    }

    // 画像をシャッフルして必要な数だけ取得
    const shuffled = [...images].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count).map((image, index) => ({
      id: image.id || index,
      src: `file://${image.path}`,
      setId: this.currentImageSet
    }));
  }
}

export default new ImageManager();
