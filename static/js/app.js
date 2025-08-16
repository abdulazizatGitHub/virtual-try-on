// Virtual Try-On Application JavaScript
class VirtualTryOnApp {
    constructor() {
        this.personFile = null;
        this.clothingFile = null;
        this.personImage = null;
        this.clothingImage = null;
        this.tryonResult = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupDragAndDrop();
        this.updateTryOnButton();
    }

    setupEventListeners() {
        // File input change events
        document.getElementById('personFile').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0], 'person');
        });

        document.getElementById('clothingFile').addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0], 'clothing');
        });

        // Try-on button click
        document.getElementById('tryonBtn').addEventListener('click', () => {
            this.generateTryOn();
        });

        // Click on upload areas
        document.getElementById('personUploadArea').addEventListener('click', () => {
            document.getElementById('personFile').click();
        });

        document.getElementById('clothingUploadArea').addEventListener('click', () => {
            document.getElementById('clothingFile').click();
        });
    }

    setupDragAndDrop() {
        const uploadAreas = [
            { element: document.getElementById('personUploadArea'), type: 'person' },
            { element: document.getElementById('clothingUploadArea'), type: 'clothing' }
        ];

        uploadAreas.forEach(({ element, type }) => {
            element.addEventListener('dragover', (e) => {
                e.preventDefault();
                element.style.borderColor = '#667eea';
                element.style.backgroundColor = '#f0f4ff';
            });

            element.addEventListener('dragleave', (e) => {
                e.preventDefault();
                element.style.borderColor = '#e9ecef';
                element.style.backgroundColor = 'white';
            });

            element.addEventListener('drop', (e) => {
                e.preventDefault();
                element.style.borderColor = '#e9ecef';
                element.style.backgroundColor = 'white';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.handleFileSelect(files[0], type);
                }
            });
        });
    }

    handleFileSelect(file, type) {
        if (!file) return;

        // Validate file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            this.showToast('Please select a valid image file (JPG, PNG, BMP)', 'error');
            return;
        }

        // Validate file size (max 10MB)
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            this.showToast('File size must be less than 10MB', 'error');
            return;
        }

        // Store file reference
        if (type === 'person') {
            this.personFile = file;
        } else {
            this.clothingFile = file;
        }

        // Preview image
        this.previewImage(file, type);
        this.updateTryOnButton();
    }

    previewImage(file, type) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                if (type === 'person') {
                    this.personImage = img;
                    this.showPreview(img, 'personPreview', 'personPlaceholder');
                } else {
                    this.clothingImage = img;
                    this.showPreview(img, 'clothingPreview', 'clothingPlaceholder');
                }
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    showPreview(img, previewId, placeholderId) {
        const preview = document.getElementById(previewId);
        const placeholder = document.getElementById(placeholderId);
        
        preview.src = img.src;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
    }

    updateTryOnButton() {
        const tryonBtn = document.getElementById('tryonBtn');
        const isEnabled = this.personFile && this.clothingFile;
        
        tryonBtn.disabled = !isEnabled;
        
        if (isEnabled) {
            tryonBtn.style.opacity = '1';
        } else {
            tryonBtn.style.opacity = '0.6';
        }
    }

    async generateTryOn() {
        if (!this.personFile || !this.clothingFile) {
            this.showToast('Please upload both a person photo and clothing image', 'error');
            return;
        }

        // Show loading state
        this.showLoading(true);
        this.hideResults();

        try {
            // Create FormData for API call
            const formData = new FormData();
            formData.append('person_image', this.personFile);
            formData.append('clothing_image', this.clothingFile);

            // Make API call
            const response = await fetch('/api/v1/tryon', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.success) {
                // Display results
                this.displayResults(result.data);
                this.showToast('Try-on generated successfully!', 'success');
            } else {
                throw new Error(result.message || 'Failed to generate try-on');
            }

        } catch (error) {
            console.error('Error generating try-on:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayResults(result) {
        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';

        // Set images
        if (this.personImage) {
            document.getElementById('originalPerson').src = this.personImage.src;
        }
        if (this.clothingImage) {
            document.getElementById('originalClothing').src = this.clothingImage.src;
        }

        // Set try-on result - extract filename from result_path or use result_url
        const tryonResultImg = document.getElementById('tryonResult');
        let filename;
        if (result.filename) {
            filename = result.filename;
        } else if (result.result_path) {
            // Extract filename from full path
            filename = result.result_path.split(/[\\/]/).pop();
        } else if (result.result_url) {
            // Extract filename from URL
            filename = result.result_url.split('/').pop();
        } else {
            console.error('No filename found in result:', result);
            this.showToast('Error: Could not determine result filename', 'error');
            return;
        }
        
        tryonResultImg.src = `/results/${filename}`;
        this.tryonResult = filename;

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    hideResults() {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'none';
    }

    showLoading(show) {
        const tryonBtn = document.getElementById('tryonBtn');
        const loadingSpinner = document.getElementById('loadingSpinner');

        if (show) {
            tryonBtn.style.display = 'none';
            loadingSpinner.style.display = 'block';
        } else {
            tryonBtn.style.display = 'block';
            loadingSpinner.style.display = 'none';
        }
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = document.createElement('i');
        icon.className = this.getToastIcon(type);
        
        const text = document.createElement('span');
        text.textContent = message;
        
        toast.appendChild(icon);
        toast.appendChild(text);
        toastContainer.appendChild(toast);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }

    getToastIcon(type) {
        switch (type) {
            case 'success':
                return 'fas fa-check-circle';
            case 'error':
                return 'fas fa-exclamation-circle';
            default:
                return 'fas fa-info-circle';
        }
    }

    downloadResult() {
        if (!this.tryonResult) {
            this.showToast('No result to download', 'error');
            return;
        }

        const link = document.createElement('a');
        link.href = `/results/${this.tryonResult}`;
        link.download = `tryon_result_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        this.showToast('Download started!', 'success');
    }

    shareResult() {
        if (!this.tryonResult) {
            this.showToast('No result to share', 'error');
            return;
        }

        // Check if Web Share API is supported
        if (navigator.share) {
            navigator.share({
                title: 'My Virtual Try-On Result',
                text: 'Check out my virtual try-on result!',
                url: window.location.href
            }).then(() => {
                this.showToast('Shared successfully!', 'success');
            }).catch((error) => {
                console.log('Error sharing:', error);
                this.showToast('Failed to share', 'error');
            });
        } else {
            // Fallback: copy URL to clipboard
            const url = `${window.location.origin}/results/${this.tryonResult}`;
            navigator.clipboard.writeText(url).then(() => {
                this.showToast('Result URL copied to clipboard!', 'success');
            }).catch(() => {
                this.showToast('Failed to copy URL', 'error');
            });
        }
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VirtualTryOnApp();
});

// Global functions for onclick handlers
function downloadResult() {
    if (window.virtualTryOnApp) {
        window.virtualTryOnApp.downloadResult();
    }
}

function shareResult() {
    if (window.virtualTryOnApp) {
        window.virtualTryOnApp.shareResult();
    }
}
