'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import React from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Ruler, Weight, Target, Camera, X, Upload, AlertCircle } from 'lucide-react';
import BodySilhouette from '@/components/BodySilhouette';
import { BodyMeasurements } from '@/types';
import { uploadImage, checkBackendHealth, UploadResponse } from '@/services/imageUpload';

export default function InputPage() {
  const router = useRouter();
  const [measurements, setMeasurements] = useState<BodyMeasurements>({
    height: 0,
    weight: 0,
    wingspan: 0,
    shoulderWidth: 0,
    waist: 0,
    hip: 0
  });
  const [userPhoto, setUserPhoto] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [backendConnected, setBackendConnected] = useState<boolean | null>(null);

  const handleMeasurementChange = (key: keyof BodyMeasurements, value: number) => {
    const numericValue = Number(value);
    console.log(`Updating ${key} to:`, numericValue, 'Type:', typeof numericValue);
    setMeasurements(prev => ({
      ...prev,
      [key]: numericValue
    }));
  };

  // Check for user photo and extracted measurements on component mount
  React.useEffect(() => {
    const photo = localStorage.getItem('userPhoto');
    if (photo) {
      setUserPhoto(photo);
    }
    
    // Check for extracted measurements from photo processing
    const extractedData = localStorage.getItem('extractedMeasurements');
    if (extractedData) {
      try {
        const parsedData = JSON.parse(extractedData);
        console.log('Extracted measurements from photo:', parsedData);
        
        // If we have extracted measurements, populate the form
        if (parsedData.extracted_measurements) {
          const extractedMeasurements = parsedData.extracted_measurements;
          console.log('Auto-populating measurements:', extractedMeasurements);
          
          setMeasurements({
            height: extractedMeasurements.height || 0,
            weight: extractedMeasurements.weight || 0,
            wingspan: extractedMeasurements.wingspan || 0,
            shoulderWidth: extractedMeasurements.shoulderWidth || 0,
            waist: extractedMeasurements.waist || 0,
            hip: extractedMeasurements.hip || 0
          });
        }
      } catch (error) {
        console.error('Error parsing extracted measurements:', error);
      }
    }
    
    // Check backend health
    checkBackendHealth().then(setBackendConnected);
  }, []);

  const validateMeasurements = (measurements: BodyMeasurements): boolean => {
    const validationRanges = {
      height: { min: 100, max: 250 },
      weight: { min: 30, max: 200 },
      wingspan: { min: 100, max: 250 },
      shoulderWidth: { min: 30, max: 80 },
      waist: { min: 50, max: 150 },
      hip: { min: 60, max: 160 }
    };

    // Check if any measurement is out of range
    for (const [key, value] of Object.entries(measurements)) {
      const field = key as keyof BodyMeasurements;
      const range = validationRanges[field];
      
      if (value > 0 && (value < range.min || value > range.max)) {
        return false; // Found an out-of-range measurement
      }
    }
    
    return true; // All measurements are within range
  };

  const handleSubmit = async () => {
    console.log('handleSubmit called, userPhoto:', !!userPhoto);
    setUploadError(null);
    
    // If user has a photo, upload it to the backend first
    if (userPhoto) {
      console.log('Starting image upload...');
      setIsUploading(true);
      
      try {
        const uploadResult: UploadResponse = await uploadImage(userPhoto);
        
        if (uploadResult.success) {
          console.log('Image uploaded successfully:', uploadResult.data);
          
          // Store the backend response data for later use
          localStorage.setItem('uploadedImageData', JSON.stringify(uploadResult.data));
        } else {
          setUploadError(uploadResult.error || 'Failed to upload image');
          setIsUploading(false);
          return; // Don't proceed if upload failed
        }
      } catch (error) {
        setUploadError('Unexpected error during upload');
        setIsUploading(false);
        return;
      }
      
      setIsUploading(false);
    }
    
    // Store measurements in localStorage for the loading and results pages
    localStorage.setItem('userMeasurements', JSON.stringify(measurements));
    
    // Check if measurements are within expected ranges
    if (validateMeasurements(measurements)) {
      router.push('/loading');
    } else {
      router.push('/validation');
    }
  };

  const handleClearPhoto = () => {
    localStorage.removeItem('userPhoto');
    setUserPhoto(null);
  };

  const isFormValid = (
    (Number(measurements.height) > 0 && 
     Number(measurements.weight) > 0 && 
     Number(measurements.wingspan) > 0) || 
    userPhoto
  );
  
  // Debug logging
  console.log('Form validation debug:', {
    height: measurements.height,
    weight: measurements.weight,
    wingspan: measurements.wingspan,
    userPhoto,
    isFormValid
  });

  return (
    <div className="min-h-screen py-8 px-4 relative overflow-hidden">
      {/* Bold Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-radial from-neon-pink/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-radial from-neon-blue/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-neon-green/10 to-transparent rounded-full blur-2xl"></div>
      </div>
      
      <div className="max-w-6xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <button
            onClick={() => router.back()}
            className="flex items-center gap-2 text-neon-blue hover:text-neon-green transition-colors duration-300"
          >
            <ArrowLeft size={20} />
            <span className="font-oswald">Back</span>
          </button>
          <h1 className="text-4xl md:text-5xl font-oswald gradient-text font-black">
            Your Measurements
          </h1>
          {userPhoto && (
            <button
              onClick={() => router.push('/photo')}
              className="flex items-center gap-2 text-neon-pink hover:text-neon-blue transition-colors duration-300"
            >
              <Camera size={20} />
              <span className="font-oswald">Retake Photo</span>
            </button>
          )}
          <div className="w-20"></div> {/* Spacer for centering */}
        </motion.div>

        {/* Backend Connection Status */}
        {backendConnected === false && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-red-900/20 border border-red-500 rounded-xl flex items-center gap-3"
          >
            <AlertCircle className="text-red-400" size={20} />
            <div>
              <p className="text-red-400 font-oswald font-bold">Backend Disconnected</p>
              <p className="text-red-300 text-sm font-oswald">
                Image upload unavailable. Make sure Flask server is running on localhost:5000
              </p>
            </div>
          </motion.div>
        )}

        {/* Upload Error Display */}
        {uploadError && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-red-900/20 border border-red-500 rounded-xl flex items-center gap-3"
          >
            <AlertCircle className="text-red-400" size={20} />
            <div>
              <p className="text-red-400 font-oswald font-bold">Upload Failed</p>
              <p className="text-red-300 text-sm font-oswald">{uploadError}</p>
            </div>
          </motion.div>
        )}

        {/* Photo Preview Section */}
        {userPhoto && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8 card-green p-6"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <img
                  src={userPhoto}
                  alt="Your photo"
                  className="w-20 h-20 object-cover rounded-xl border-2 border-neon-green"
                />
                <div>
                  <h3 className="text-xl font-oswald font-black text-neon-green mb-2">
                    Photo Captured
                  </h3>
                  <p className="text-text-secondary font-oswald">
                    {backendConnected === false 
                      ? "Will be processed locally (backend offline)"
                      : "Will be uploaded for enhanced analysis"
                    }
                  </p>
                </div>
              </div>
              <button
                onClick={handleClearPhoto}
                className="btn-secondary px-4 py-2 flex items-center gap-2 hover:bg-red-600 hover:border-red-500 transition-all duration-300"
              >
                <X size={16} />
                <span className="font-oswald">Clear Photo</span>
              </button>
            </div>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Body Silhouette */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="card-blue p-10"
          >
            <div className="text-center mb-6">
              <Target className="mx-auto mb-3 text-neon-blue" size={32} />
              <h2 className="text-3xl font-oswald font-black text-neon-blue mb-3">
                Interactive Body Map
              </h2>
              <p className="text-text-secondary text-lg font-oswald">
                Click on different body parts to enter your measurements
              </p>
            </div>
            
            <BodySilhouette
              measurements={measurements}
              onMeasurementChange={handleMeasurementChange}
            />
          </motion.div>

          {/* Manual Input Form */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="card-pink p-10"
          >
            <div className="text-center mb-6">
              <Ruler className="mx-auto mb-3 text-neon-green" size={32} />
              <h2 className="text-3xl font-oswald font-black text-neon-pink mb-3">
                Manual Input
              </h2>
              <p className="text-text-secondary text-lg font-oswald">
                Or enter your measurements directly below
              </p>
            </div>

            <div className="space-y-6">
              {/* Height */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Height (cm)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.height || ''}
                    onChange={(e) => handleMeasurementChange('height', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your height"
                  />
                  <Ruler className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>

              {/* Weight */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Weight (kg)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.weight || ''}
                    onChange={(e) => handleMeasurementChange('weight', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your weight"
                  />
                  <Weight className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>

              {/* Wingspan */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Wingspan (cm)
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.wingspan || ''}
                    onChange={(e) => handleMeasurementChange('wingspan', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your wingspan"
                  />
                  <Target className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>

              {/* Shoulder Width */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Shoulder Width (cm) - Optional
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.shoulderWidth || ''}
                    onChange={(e) => handleMeasurementChange('shoulderWidth', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your shoulder width"
                  />
                  <Ruler className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>

              {/* Waist */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Waist (cm) - Optional
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.waist || ''}
                    onChange={(e) => handleMeasurementChange('waist', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your waist measurement"
                  />
                  <Ruler className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>

              {/* Hip */}
              <div>
                <label className="block text-sm font-oswald font-bold text-text-secondary mb-3">
                  Hip (cm) - Optional
                </label>
                <div className="relative">
                  <input
                    type="number"
                    value={measurements.hip || ''}
                    onChange={(e) => handleMeasurementChange('hip', Number(e.target.value) || 0)}
                    className="input-field w-full pl-10"
                    placeholder="Enter your hip measurement"
                  />
                  <Ruler className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <motion.button
              onClick={handleSubmit}
              disabled={!isFormValid || isUploading}
              className={`w-full mt-8 py-6 px-8 rounded-xl font-oswald font-black text-xl transition-all duration-500 flex items-center justify-center gap-3 ${
                isFormValid && !isUploading
                  ? 'btn-accent neon-glow-green hover:neon-glow-pink'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'
              }`}
              whileHover={isFormValid && !isUploading ? { scale: 1.02 } : {}}
              whileTap={isFormValid && !isUploading ? { scale: 0.98 } : {}}
            >
              {isUploading ? (
                <>
                  <Upload className="animate-pulse" size={20} />
                  Uploading Image...
                </>
              ) : isFormValid ? (
                userPhoto && measurements.height === 0 ? 'See My Match (Photo Only)' : 'See My Match'
              ) : (
                'Enter Required Measurements'
              )}
            </motion.button>

            {/* Progress Indicator */}
            <div className="mt-6">
              <div className="flex justify-between text-sm text-gray-400 mb-2 font-oswald">
                <span>Progress</span>
                <span>
                  {userPhoto 
                    ? "Photo provided" 
                    : `${[measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length}/3 measurements`
                  }
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-neon-blue to-neon-green h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{
                    width: userPhoto 
                      ? "100%" 
                      : `${([measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length / 3) * 100}%`
                  }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </motion.div>
        </div>

        {/* Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-12 card p-6"
        >
          <h3 className="text-xl font-oswald font-bold text-neon-gold mb-4">
            Measurement Tips
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-400 font-oswald">
            <div>
              <strong className="text-neon-blue">Height:</strong> Measure without shoes, standing straight against a wall
            </div>
            <div>
              <strong className="text-neon-green">Weight:</strong> Use a digital scale, preferably in the morning
            </div>
            <div>
              <strong className="text-neon-gold">Wingspan:</strong> Stretch arms horizontally, measure fingertip to fingertip
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
