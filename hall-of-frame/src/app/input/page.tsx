'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import React from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Ruler, Weight, Target, Camera } from 'lucide-react';
import BodySilhouette from '@/components/BodySilhouette';
import { BodyMeasurements } from '@/types';

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

  const handleMeasurementChange = (key: keyof BodyMeasurements, value: number) => {
    setMeasurements(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Check for user photo on component mount
  React.useEffect(() => {
    const photo = localStorage.getItem('userPhoto');
    if (photo) {
      setUserPhoto(photo);
    }
  }, []);

  const handleSubmit = () => {
    // Store measurements in localStorage for the loading and results pages
    localStorage.setItem('userMeasurements', JSON.stringify(measurements));
    router.push('/loading');
  };

  const isFormValid = (measurements.height > 0 && measurements.weight > 0 && measurements.wingspan > 0) || userPhoto;

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

        {/* Photo Preview Section */}
        {userPhoto && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8 card-green p-6"
          >
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
                  This photo will help with more accurate analysis
                </p>
              </div>
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
                    onChange={(e) => handleMeasurementChange('height', Number(e.target.value))}
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
                    onChange={(e) => handleMeasurementChange('weight', Number(e.target.value))}
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
                    onChange={(e) => handleMeasurementChange('wingspan', Number(e.target.value))}
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
                    onChange={(e) => handleMeasurementChange('shoulderWidth', Number(e.target.value))}
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
                    onChange={(e) => handleMeasurementChange('waist', Number(e.target.value))}
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
                    onChange={(e) => handleMeasurementChange('hip', Number(e.target.value))}
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
              disabled={!isFormValid}
              className={`w-full mt-8 py-6 px-8 rounded-xl font-oswald font-black text-xl transition-all duration-500 ${
                isFormValid
                  ? 'btn-accent neon-glow-green hover:neon-glow-pink'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'
              }`}
              whileHover={isFormValid ? { scale: 1.02 } : {}}
              whileTap={isFormValid ? { scale: 0.98 } : {}}
            >
              {isFormValid 
                ? (userPhoto && measurements.height === 0 ? 'See My Match (Photo Only)' : 'See My Match')
                : 'Enter Required Measurements'
              }
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
            💡 Measurement Tips
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
