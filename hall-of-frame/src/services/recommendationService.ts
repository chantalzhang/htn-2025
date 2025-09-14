import axios from 'axios';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';

export interface BackendRecommendationRequest {
  gender: string;
  height: number;
  weight: number;
  wingspan: number;
  shoulderWidth: number;
  waist: number;
  hip: number;
}

export interface BackendRecommendationResponse {
  success: boolean;
  data?: {
    recommendation: {
      top_sport: {
        sport: string;
        stats: {
          strength: number;
          agility: number;
          endurance: number;
          power: number;
          speed: number;
          flexibility: number;
          coordination: number;
          balance: number;
        };
        description: string;
        athlete_count: number;
      };
      all_sports: Array<{
        sport: string;
        stats: {
          strength: number;
          agility: number;
          endurance: number;
          power: number;
          speed: number;
          flexibility: number;
          coordination: number;
          balance: number;
        };
        description: string;
        athlete_count: number;
      }>;
    };
    similar_athletes: Array<{
      name: string;
      sport: string;
      gender_emoji: string;
      measurements: {
        height: number | string;
        weight: number | string;
        wingspan: number | string;
        shoulderWidth: number | string;
        waist: number | string;
        hip: number | string;
      };
    }>;
  };
  error?: string;
}

/**
 * Get sport recommendation and similar athlete from backend ML models
 */
export async function getRecommendation(
  measurements: BackendRecommendationRequest
): Promise<BackendRecommendationResponse> {
  try {
    console.log('Calling backend recommendation API with:', measurements);
    
    const response = await axios.post(`${BACKEND_URL}/recommend`, measurements, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    });
    
    console.log('Backend recommendation response:', response.data);
    return response.data as BackendRecommendationResponse;
    
  } catch (error) {
    console.error('Backend recommendation error:', error);
    
    if (axios.isAxiosError(error)) {
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
 * Check if backend recommendation service is available
 */
export async function checkRecommendationService(): Promise<boolean> {
  try {
    const response = await axios.get(`${BACKEND_URL}/`, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch (error) {
    console.error('Backend health check failed:', error);
    return false;
  }
}
