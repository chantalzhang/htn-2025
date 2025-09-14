'use client';

import { motion } from 'framer-motion';
import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Camera, Upload, SkipForward, RotateCcw, Loader } from 'lucide-react';
import { uploadImage, UploadResponse } from '@/services/imageUpload';

export default function PhotoPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState<string | null>(null);
  const [height, setHeight] = useState<string>('');

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        } 
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsCapturing(true);
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Unable to access camera. Please check permissions or try uploading an image instead.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCapturing(false);
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current) {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const context = canvas.getContext('2d');
      
      if (context) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0);
        
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        setCapturedImage(imageData);
        stopCamera();
      }
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setUploadedImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleContinue = async () => {
    const imageData = capturedImage || uploadedImage;
    if (!imageData) return;

    // Check if height is provided
    if (!height || height.trim() === '') {
      alert('Please enter your height before continuing.');
      return;
    }

    const heightValue = parseFloat(height);
    if (isNaN(heightValue) || heightValue < 100 || heightValue > 250) {
      alert('Please enter a valid height between 100 and 250 cm.');
      return;
    }

    setIsProcessing(true);
    setProcessingError(null);

    try {
      console.log('Uploading image to backend for measurement extraction...');
      const uploadResult: UploadResponse = await uploadImage(imageData);
      
      if (uploadResult.success) {
        console.log('Image processed successfully:', uploadResult.data);
        
        // Store the image, processing results, and height
        localStorage.setItem('userPhoto', imageData);
        localStorage.setItem('extractedMeasurements', JSON.stringify(uploadResult.data));
        localStorage.setItem('userHeight', heightValue.toString());
        
        // Navigate to input page
        router.push('/input');
      } else {
        setProcessingError(uploadResult.error || 'Failed to process image');
        setIsProcessing(false);
      }
    } catch (error) {
      console.error('Error processing image:', error);
      setProcessingError('Unexpected error occurred while processing image');
      setIsProcessing(false);
    }
  };

  const handleSkip = () => {
    router.push('/input');
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setUploadedImage(null);
    setHeight('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen py-8 px-4 relative overflow-hidden">
      {/* Bold Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-radial from-neon-pink/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-radial from-neon-blue/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-neon-green/10 to-transparent rounded-full blur-2xl"></div>
      </div>
      
      <div className="max-w-4xl mx-auto relative z-10">
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
            Capture Your Frame
          </h1>
          <div className="w-20"></div> {/* Spacer for centering */}
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Photo Capture Section */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="card-blue p-10"
          >
            <div className="text-center mb-6">
              <Camera className="mx-auto mb-4 text-neon-blue" size={48} />
              <h2 className="text-3xl font-oswald font-black text-neon-blue mb-3">
                Take a Photo
              </h2>
              <p className="text-text-secondary text-lg font-oswald">
                Position yourself in good lighting for the best results
              </p>
            </div>

            {/* Camera Preview */}
            <div className="relative mb-6">
              {!capturedImage && !uploadedImage && (
                <div className="aspect-video bg-gray-800 rounded-xl flex items-center justify-center border-2 border-dashed border-gray-600">
                  {isCapturing ? (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="w-full h-full object-cover rounded-xl"
                    />
                  ) : (
                    <div className="text-center">
                      <Camera className="mx-auto mb-4 text-gray-400" size={64} />
                      <p className="text-gray-400 font-oswald">Camera preview will appear here</p>
                    </div>
                  )}
                </div>
              )}

              {/* Captured/Uploaded Image Display */}
              {(capturedImage || uploadedImage) && (
                <div className="relative">
                  <img
                    src={capturedImage || uploadedImage || ''}
                    alt="Captured photo"
                    className="w-full aspect-video object-cover rounded-xl"
                  />
                  <button
                    onClick={handleRetake}
                    className="absolute top-4 right-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors duration-300"
                  >
                    <RotateCcw size={20} />
                  </button>
                </div>
              )}

              {/* Hidden canvas for photo capture */}
              <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Camera Controls */}
            {!capturedImage && !uploadedImage && (
              <div className="space-y-4">
                {!isCapturing ? (
                  <motion.button
                    onClick={startCamera}
                    className="w-full btn-primary py-4 px-6 text-xl font-oswald font-black"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Camera className="inline-block mr-3" size={24} />
                    Start Camera
                  </motion.button>
                ) : (
                  <motion.button
                    onClick={capturePhoto}
                    className="w-full btn-accent py-4 px-6 text-xl font-oswald font-black"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Camera className="inline-block mr-3" size={24} />
                    Capture Photo
                  </motion.button>
                )}
              </div>
            )}
          </motion.div>

          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="card-pink p-10"
          >
            <div className="text-center mb-6">
              <Upload className="mx-auto mb-4 text-neon-pink" size={48} />
              <h2 className="text-3xl font-oswald font-black text-neon-pink mb-3">
                Upload Image
              </h2>
              <p className="text-text-secondary text-lg font-oswald">
                Choose an existing photo from your device
              </p>
            </div>

            <div className="space-y-6">
              {/* File Upload */}
              <div className="border-2 border-dashed border-gray-600 rounded-xl p-8 text-center hover:border-neon-pink transition-colors duration-300">
                <Upload className="mx-auto mb-4 text-gray-400" size={48} />
                <p className="text-gray-400 font-oswald mb-4">
                  Drag and drop an image here, or click to browse
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label
                  htmlFor="file-upload"
                  className="btn-secondary cursor-pointer inline-block"
                >
                  <Upload className="inline-block mr-2" size={20} />
                  Choose File
                </label>
              </div>

              {/* File Requirements */}
              <div className="text-sm text-gray-400 font-oswald">
                <p className="mb-2"><strong>Supported formats:</strong> JPG, PNG, WebP</p>
                <p><strong>Max size:</strong> 10MB</p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Height Input Field - Only show when photo is captured/uploaded */}
        {(capturedImage || uploadedImage) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 max-w-md mx-auto"
          >
            <div className="card-green p-6">
              <h3 className="text-2xl font-oswald font-black text-neon-green mb-4 text-center">
                Enter Your Height
              </h3>
              <p className="text-text-secondary text-center font-oswald mb-6">
                We need your height to provide accurate measurements
              </p>
              <div className="space-y-4">
                <div>
                  <label htmlFor="height-input" className="block text-sm font-oswald font-bold text-neon-green mb-2">
                    Height (cm)
                  </label>
                  <input
                    id="height-input"
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(e.target.value)}
                    placeholder="Enter your height in centimeters"
                    className="w-full px-4 py-3 bg-gray-800 border-2 border-gray-600 rounded-xl text-white font-oswald focus:border-neon-green focus:outline-none transition-colors duration-300"
                    min="100"
                    max="250"
                    step="0.1"
                  />
                </div>
                <div className="text-xs text-gray-400 font-oswald text-center">
                  <p>Example: 175 cm (5'9")</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {/* Processing Error Display */}
        {processingError && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 p-4 bg-red-900/20 border border-red-500 rounded-xl"
          >
            <p className="text-red-400 font-oswald font-bold">Processing Failed</p>
            <p className="text-red-300 text-sm font-oswald">{processingError}</p>
          </motion.div>
        )}

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-12 flex flex-col sm:flex-row gap-4 justify-center"
        >
          {(capturedImage || uploadedImage) && (
            <motion.button
              onClick={handleContinue}
              disabled={isProcessing || !height || height.trim() === ''}
              className={`py-4 px-8 text-xl font-oswald font-black flex items-center justify-center gap-3 ${
                isProcessing || !height || height.trim() === ''
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                  : 'btn-primary'
              }`}
              whileHover={!isProcessing && height && height.trim() !== '' ? { scale: 1.05 } : {}}
              whileTap={!isProcessing && height && height.trim() !== '' ? { scale: 0.95 } : {}}
            >
              {isProcessing ? (
                <>
                  <Loader className="animate-spin" size={24} />
                  Processing Image...
                </>
              ) : !height || height.trim() === '' ? (
                'Enter Height to Continue'
              ) : (
                'Continue with Photo'
              )}
            </motion.button>
          )}
          
          <motion.button
            onClick={handleSkip}
            disabled={isProcessing}
            className={`py-4 px-8 text-xl font-oswald font-black ${
              isProcessing ? 'opacity-50 cursor-not-allowed' : 'btn-secondary'
            }`}
            whileHover={!isProcessing ? { scale: 1.05 } : {}}
            whileTap={!isProcessing ? { scale: 0.95 } : {}}
          >
            <SkipForward className="inline-block mr-3" size={24} />
            Skip to Manual Input
          </motion.button>
        </motion.div>

        {/* Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-12 card-green p-8"
        >
          <h3 className="text-2xl font-oswald font-black text-neon-green mb-6">
            📸 Photo Tips for Best Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-text-secondary font-oswald">
            <div>
              <strong className="text-neon-blue">Lighting:</strong> Stand in good, even lighting (avoid harsh shadows)
            </div>
            <div>
              <strong className="text-neon-pink">Position:</strong> Stand straight with arms at your sides
            </div>
            <div>
              <strong className="text-neon-green">Background:</strong> Use a plain, contrasting background
            </div>
            <div>
              <strong className="text-neon-orange">Clothing:</strong> Wear form-fitting clothes for accurate measurements
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
