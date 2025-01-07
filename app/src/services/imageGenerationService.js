import sessionManager from '../utils/sessionManager';

class ImageGenerationService {
    constructor() {
        this.baseUrl = 'http://host.docker.internal:8000';
    }

    async generateImages(prompt, negativePrompt = null, baseVector = null, generation = 0, numImages = 4) {
        try {
            console.log('Sending request to:', `${this.baseUrl}/generate`);
            
            const requestPayload = {
                prompt,
                negative_prompt: negativePrompt,
                base_vector: baseVector,
                num_images: numImages,
                generation,
                session_id: sessionManager.sessionId,
                interaction_mode: sessionManager.interactionMode  // インタラクションモードを追加
            };
            
            console.log('Request payload:', requestPayload);

            const controller = new AbortController();
            const timeout = 300000;
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(`${this.baseUrl}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestPayload),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                console.error('Server error:', errorData);
                throw new Error(errorData.detail || 'Failed to generate images');
            }

            const data = await response.json();
            console.log('Received response:', data);
            return data;

        } catch (error) {
            console.error('Error in generateImages:', error);
            throw error;
        }
    }

    getImageUrl(sessionId, step, filename) {
        const url = `${this.baseUrl}/session-data/${sessionId}/step_${step}/${filename}`;
        console.log('Generated image URL:', url);
        return url;
    }

    saveImage(image, filepath) {
        image.save(filepath);
    }

    latent_to_base64(latent) {
        if (!latent) return null;
        const numpy_data = latent.cpu().numpy().astype(np.float16);
        const bytes_data = numpy_data.tobytes();
        const base64_str = base64.b64encode(bytes_data).decode('utf-8');
        return base64_str;
    }

    base64_to_latent(base64_str) {
        if (!base64_str) return null;
        const latent_bytes = base64.b64decode(base64_str);
        const latent_np = np.frombuffer(latent_bytes, dtype=np.float16).copy();
        const tensor = torch.from_numpy(latent_np).to(device=this.device, dtype=torch.float16);
        return tensor;
    }
}

export default new ImageGenerationService();
