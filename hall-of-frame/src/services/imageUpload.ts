import axios from 'axios';

// Backend URL - adjust this to match your Flask server
const BACKEND_URL = 'http://localhost:5000';

export interface UploadResponse {
  success: boolean;
  data?: {
    message: string;
    filename: string;
    file_path: string;
    file_size: number;
  };
  error?: string;
}

/**
 * Convert base64 data URL to File object
 */
function dataURLtoFile(dataurl: string, filename: string): File {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/jpeg';
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  
  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }
  
  return new File([u8arr], filename, { type: mime });
}

/**
 * Upload image to Flask backend
 * @param imageData - Base64 data URL from localStorage or File object
 * @param filename - Optional filename (will generate one if not provided)
 */
export async function uploadImage(
  imageData: string | File, 
  filename?: string
): Promise<UploadResponse> {
  try {
    let file: File;
    
    // Convert base64 to File if needed
    if (typeof imageData === 'string') {
      const defaultFilename = filename || `upload_${Date.now()}.jpg`;
      file = dataURLtoFile(imageData, defaultFilename);
    } else {
      file = imageData;
    }
    console.log("in the uploadimage function")
    // Create FormData
    const formData = new FormData();
    formData.append('image', file);
    
    // Send request to Flask backend
    const response = await axios.post(`${BACKEND_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 30000, // 30 second timeout
    });
    console.log(response)
    return response.data as UploadResponse;
    
  } catch (error) {
    console.error('Image upload error:', error);
    
    if (axios.isAxiosError(error)) {
      // Handle axios-specific errors
      if (error.response) {
        // Server responded with error status
        return {
          success: false,
          error: error.response.data?.error || `Server error: ${error.response.status}`
        };
      } else if (error.request) {
        // Request was made but no response received
        return {
          success: false,
          error: 'No response from server. Please check if the backend is running.'
        };
      }
    }
    
    // Generic error
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred'
    };
  }
}

/**
 * Check if backend is available
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${BACKEND_URL}/`, {
      timeout: 5000
    });
    return response.data?.status === 'healthy';
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
}
