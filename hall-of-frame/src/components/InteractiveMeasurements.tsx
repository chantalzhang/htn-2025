'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { Target, Ruler } from 'lucide-react';
import { BodyMeasurements } from '@/types';
import BodyMap from './BodyMap';
import MeasurementForm from './MeasurementForm';

interface InteractiveMeasurementsProps {
  measurements: BodyMeasurements;
  onMeasurementChange: (key: keyof BodyMeasurements, value: number) => void;
}

export default function InteractiveMeasurements({
  measurements,
  onMeasurementChange
}: InteractiveMeasurementsProps) {
  const [activeField, setActiveField] = useState<keyof BodyMeasurements | null>(null);
  const [hoveredRegion, setHoveredRegion] = useState<string | null>(null);

  const handleFieldFocus = (field: keyof BodyMeasurements) => {
    setActiveField(field);
  };

  const handleFieldBlur = () => {
    setActiveField(null);
  };

  const handleRegionHover = (region: string | null) => {
    setHoveredRegion(region);
  };

  const handleRegionClick = (field: keyof BodyMeasurements) => {
    setActiveField(field);
    // Smooth animated focus transition
    setTimeout(() => {
      const inputElement = document.querySelector(`input[data-field="${field}"]`) as HTMLInputElement;
      if (inputElement) {
        inputElement.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center',
          inline: 'nearest'
        });
        setTimeout(() => {
          inputElement.focus();
        }, 300);
      }
    }, 100);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
      {/* Interactive Body Map */}
      <motion.div
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8 }}
        className="relative"
      >
        {/* Card Container */}
        <div className="card-blue p-8 relative overflow-hidden">
          {/* Background Effects */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-4 right-4 w-32 h-32 bg-gradient-radial from-neon-blue/10 to-transparent rounded-full blur-xl"></div>
            <div className="absolute bottom-4 left-4 w-24 h-24 bg-gradient-radial from-neon-green/10 to-transparent rounded-full blur-lg"></div>
          </div>

          <div className="text-center mb-6 relative z-10">
            <motion.div
              animate={{ 
                rotate: activeField ? 360 : 0,
                scale: hoveredRegion ? 1.1 : 1
              }}
              transition={{ duration: 0.5 }}
            >
              <Target className="mx-auto mb-3 text-neon-blue" size={32} />
            </motion.div>
            <h2 className="text-3xl font-oswald font-black text-neon-blue mb-3">
              Interactive Body Map
            </h2>
            <p className="text-text-secondary text-lg font-oswald">
              {hoveredRegion 
                ? "Click the highlighted region to measure" 
                : activeField 
                  ? `Measuring: ${activeField.replace(/([A-Z])/g, ' $1').toLowerCase()}`
                  : "Hover over body parts or focus on inputs to see connections"
              }
            </p>
          </div>
          
          <div className="relative z-10">
            <BodyMap
              measurements={measurements}
              activeField={activeField}
              hoveredRegion={hoveredRegion}
              onRegionHover={handleRegionHover}
              onRegionClick={handleRegionClick}
            />
          </div>

          {/* Connection Lines Animation */}
          {activeField && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 pointer-events-none"
            >
              <svg className="w-full h-full">
                <motion.path
                  d="M 50% 50% Q 75% 25% 100% 50%"
                  stroke="url(#connectionGradient)"
                  strokeWidth="2"
                  fill="none"
                  strokeDasharray="5,5"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <defs>
                  <linearGradient id="connectionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#00d4ff" stopOpacity="0" />
                    <stop offset="50%" stopColor="#00ff88" stopOpacity="1" />
                    <stop offset="100%" stopColor="#ff0080" stopOpacity="0" />
                  </linearGradient>
                </defs>
              </svg>
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Measurement Form */}
      <motion.div
        initial={{ opacity: 0, x: 50 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, delay: 0.2 }}
        className="relative"
      >
        {/* Card Container */}
        <div className="card-pink p-8 relative overflow-hidden">
          {/* Background Effects */}
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-4 left-4 w-32 h-32 bg-gradient-radial from-neon-pink/10 to-transparent rounded-full blur-xl"></div>
            <div className="absolute bottom-4 right-4 w-24 h-24 bg-gradient-radial from-neon-orange/10 to-transparent rounded-full blur-lg"></div>
          </div>

          <div className="text-center mb-6 relative z-10">
            <motion.div
              animate={{ 
                rotate: hoveredRegion ? -360 : 0,
                scale: activeField ? 1.1 : 1
              }}
              transition={{ duration: 0.5 }}
            >
              <Ruler className="mx-auto mb-3 text-neon-green" size={32} />
            </motion.div>
            <h2 className="text-3xl font-oswald font-black text-neon-pink mb-3">
              Measurement Input
            </h2>
            <p className="text-text-secondary text-lg font-oswald">
              {activeField 
                ? `Currently measuring: ${activeField.replace(/([A-Z])/g, ' $1').toLowerCase()}`
                : "Enter your measurements or interact with the body map"
              }
            </p>
          </div>

          <div className="relative z-10">
            <MeasurementForm
              measurements={measurements}
              onMeasurementChange={onMeasurementChange}
              activeField={activeField}
              onFieldFocus={handleFieldFocus}
              onFieldBlur={handleFieldBlur}
              hoveredRegion={hoveredRegion}
            />
          </div>
        </div>
      </motion.div>

    </div>
  );
}
