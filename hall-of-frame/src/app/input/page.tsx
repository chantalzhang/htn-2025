'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Ruler, Weight, Target } from 'lucide-react';
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

  const handleMeasurementChange = (key: keyof BodyMeasurements, value: number) => {
    setMeasurements(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSubmit = () => {
    // Store measurements in localStorage for the loading and results pages
    localStorage.setItem('userMeasurements', JSON.stringify(measurements));
    router.push('/loading');
  };

  const isFormValid = measurements.height > 0 && measurements.weight > 0 && measurements.wingspan > 0;

  return (
    <div className="min-h-screen bg-dark-bg py-8 px-4">
      <div className="max-w-6xl mx-auto">
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
            <span className="font-montserrat">Back</span>
          </button>
          <h1 className="text-3xl md:text-4xl font-bebas gradient-text">
            Your Measurements
          </h1>
          <div className="w-20"></div> {/* Spacer for centering */}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Body Silhouette */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            className="card p-8"
          >
            <div className="text-center mb-6">
              <Target className="mx-auto mb-3 text-neon-blue" size={32} />
              <h2 className="text-2xl font-montserrat font-bold text-neon-blue mb-2">
                Interactive Body Map
              </h2>
              <p className="text-gray-400">
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
            className="card p-8"
          >
            <div className="text-center mb-6">
              <Ruler className="mx-auto mb-3 text-neon-green" size={32} />
              <h2 className="text-2xl font-montserrat font-bold text-neon-green mb-2">
                Manual Input
              </h2>
              <p className="text-gray-400">
                Or enter your measurements directly below
              </p>
            </div>

            <div className="space-y-6">
              {/* Height */}
              <div>
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
                <label className="block text-sm font-montserrat font-medium text-gray-300 mb-2">
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
              className={`w-full mt-8 py-4 px-6 rounded-lg font-montserrat font-bold text-lg transition-all duration-300 ${
                isFormValid
                  ? 'btn-primary neon-glow-green hover:neon-glow-blue'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'
              }`}
              whileHover={isFormValid ? { scale: 1.02 } : {}}
              whileTap={isFormValid ? { scale: 0.98 } : {}}
            >
              {isFormValid ? 'See My Match' : 'Enter Required Measurements'}
            </motion.button>

            {/* Progress Indicator */}
            <div className="mt-6">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Progress</span>
                <span>
                  {[measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length}/3
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-neon-blue to-neon-green h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{
                    width: `${([measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length / 3) * 100}%`
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
          <h3 className="text-xl font-montserrat font-bold text-neon-gold mb-4">
            💡 Measurement Tips
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-400">
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
