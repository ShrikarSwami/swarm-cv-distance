"""Main application logic for the web frontend reconstruction UI.

This file coordinates the scene selection, angle selection, reconstruction,
and visualization components. It communicates with the FastAPI backend
via REST API endpoints.
"""

import { SceneSelector } from './scene-selector.js';
import { AngleSelector } from './angle-selector.js';
import { ReconstructionViewer } from './recon-viewer.js';
import { MetricsPanel } from './metrics-panel.js';

class ReconstructionApp {
    constructor() {
        this.apiBaseUrl = '/api';
        this.currentScene = null;
        this.selectedViews = [];
        this.reconstructionData = null;

        this.initApp();
    }

    async initApp() {
        try {
            // Check API health
            await this.checkApiHealth();

            // Initialize components
            this.sceneSelector = new SceneSelector(this);
            this.angleSelector = new AngleSelector(this);
            this.reconViewer = new ReconstructionViewer(this);
            this.metricsPanel = new MetricsPanel(this);

            // Load scenes
            await this.sceneSelector.loadScenes();

            // Update status
            this.updateStatus('Ready');

        } catch (error) {
            console.error('Failed to initialize app:', error);
            this.updateStatus('Error: ' + error.message);
        }
    }

    async checkApiHealth() {
        const response = await fetch(`${this.apiBaseUrl}/health`);
        const data = await response.json();
        if (data.status !== 'ok') {
            throw new Error(data.message || 'API not healthy');
        }
        return data;
    }

    updateStatus(message) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.textContent = message;
        }
        console.log('Status:', message);
    }

    async loadSceneDetails(seed) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/scene/${seed}`);
            if (!response.ok) {
                throw new Error(`Failed to load scene: ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`Error loading scene ${seed}:`, error);
            throw error;
        }
    }

    async runReconstruction(request) {
        try {
            this.updateStatus('Running reconstruction...');
            const response = await fetch(`${this.apiBaseUrl}/reconstruct`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request)
            });

            if (!response.ok) {
                throw new Error(`Reconstruction failed: ${response.statusText}`);
            }

            const result = await response.json();
            this.reconstructionData = result;
            this.updateStatus('Reconstruction complete');
            return result;

        } catch (error) {
            console.error('Error during reconstruction:', error);
            this.updateStatus('Error: ' + error.message);
            throw error;
        }
    }

    selectViews(viewIndices) {
        this.selectedViews = viewIndices;
        this.angleSelector.updateSelectionUI(viewIndices);
        this.reconViewer.updateViewSelection(viewIndices);
    }

    showReconstruction(result) {
        this.reconstructionData = result;

        // Show reconstruction view
        document.getElementById('reconstruction-view').style.display = 'block';
        document.getElementById('angle-selector').style.display = 'none';
        document.getElementById('scene-selector').style.display = 'none';

        // Update metrics panel
        this.metricsPanel.updateMetrics(result);

        // Update scene information
        this.updateSceneInfo(result);

        // Initialize 3D viewer
        this.reconViewer.render(result);
    }

    updateSceneInfo(result) {
        const sceneInfoElement = document.getElementById('scene-metadata');
        if (!sceneInfoElement) return;

        const scene = result;
        sceneInfoElement.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <label>Scene Seed:</label>
                    <span>${scene.scene_seed}</span>
                </div>
                <div class="info-item">
                    <label>True Drones:</label>
                    <span>${scene.metrics.n_true}</span>
                </div>
                <div class="info-item">
                    <label>Predicted Drones:</label>
                    <span>${scene.metrics.n_pred}</span>
                </div>
                <div class="info-item">
                    <label>Count Error:</label>
                    <span>${scene.metrics.count_err > 0 ? '+' : ''}${scene.metrics.count_err}</span>
                </div>
                <div class="info-item">
                    <label>mAP:</label>
                    <span>${scene.metrics.mAP.toFixed(3)}</span>
                </div>
                <div class="info-item">
                    <label>Median Error:</label>
                    <span>${scene.metrics.median_err_m.toFixed(3)} m</span>
                </div>
                <div class="info-item">
                    <label>Views Used:</label>
                    <span>${result.view_indices.length}/24</span>
                </div>
                <div class="info-item">
                    <label>Wall Time:</label>
                    <span>${result.wall_clock_s.toFixed(2)} s</span>
                </div>
            </div>
        `;
    }

    hideReconstruction() {
        document.getElementById('reconstruction-view').style.display = 'none';
        this.reconViewer.clear();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ReconstructionApp();
});

// Export for module usage
export { ReconstructionApp };