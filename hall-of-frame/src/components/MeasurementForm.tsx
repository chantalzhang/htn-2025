'use client';

import { motion } from 'framer-motion';
import { useRef, useEffect } from 'react';
import { Ruler, Weight, Target } from 'lucide-react';
import { BodyMeasurements } from '@/types';

interface MeasurementFormProps {
  measurements: BodyMeasurements;
  onMeasurementChange: (key: keyof BodyMeasurements, value: number) => void;
  activeField: keyof BodyMeasurements | null;
  onFieldFocus: (field: keyof BodyMeasurements) => void;
  onFieldBlur: () => void;
  hoveredRegion: string | null;
}

export default function MeasurementForm({
  measurements,
  onMeasurementChange,
  activeField,
  onFieldFocus,
  onFieldBlur,
  hoveredRegion
}: MeasurementFormProps) {
  
  // Refs for input fields to enable programmatic focus
  const inputRefs = useRef<Record<keyof BodyMeasurements, HTMLInputElement | null>>({
    height: null,
    weight: null,
    wingspan: null,
    shoulderWidth: null,
    waist: null,
    hip: null
  });

  // Map hovered regions to corresponding fields
  const regionToField: Record<string, keyof BodyMeasurements> = {
    'head': 'height',
    'torso': 'height', 
    'legs': 'height',
    'feet': 'height',
    'leftArm': 'wingspan',
    'rightArm': 'wingspan',
    'leftShoulder': 'shoulderWidth',
    'rightShoulder': 'shoulderWidth',
    'waist': 'waist',
    'hips': 'hip',
    'weight': 'weight'
  };

  // Focus input when region is clicked (handled by parent component)
  // Removed automatic focus on hover

  // Get input styling based on state
  const getInputStyling = (field: keyof BodyMeasurements) => {
    const isActive = activeField === field;
    const isFilled = measurements[field] && measurements[field]! > 0;
    const isHovered = hoveredRegion && regionToField[hoveredRegion] === field;
    
    let baseClasses = "input-field w-full pl-10 transition-all duration-300 ";
    
    if (isHovered) {
      baseClasses += "ring-4 ring-neon-pink/50 border-neon-pink shadow-lg shadow-neon-pink/25 ";
    } else if (isFilled) {
      baseClasses += "ring-2 ring-neon-green/50 border-neon-green shadow-md shadow-neon-green/20 ";
    } else if (isActive) {
      baseClasses += "ring-2 ring-neon-blue/50 border-neon-blue shadow-md shadow-neon-blue/20 ";
    }
    
    return baseClasses;
  };

  // Get label styling based on state
  const getLabelStyling = (field: keyof BodyMeasurements) => {
    const isActive = activeField === field;
    const isFilled = measurements[field] && measurements[field]! > 0;
    const isHovered = hoveredRegion && regionToField[hoveredRegion] === field;
    
    let baseClasses = "block text-sm font-oswald font-bold mb-3 transition-all duration-300 ";
    
    if (isHovered) {
      baseClasses += "text-neon-pink animate-pulse-neon ";
    } else if (isFilled) {
      baseClasses += "text-neon-green ";
    } else if (isActive) {
      baseClasses += "text-neon-blue ";
    } else {
      baseClasses += "text-text-secondary ";
    }
    
    return baseClasses;
  };

  const measurementFields = [
    {
      key: 'height' as keyof BodyMeasurements,
      label: 'Height (cm)',
      placeholder: 'Enter your height',
      icon: Ruler,
      required: true
    },
    {
      key: 'weight' as keyof BodyMeasurements,
      label: 'Weight (kg)',
      placeholder: 'Enter your weight',
      icon: Weight,
      required: true
    },
    {
      key: 'wingspan' as keyof BodyMeasurements,
      label: 'Wingspan (cm)',
      placeholder: 'Enter your wingspan',
      icon: Target,
      required: true
    },
    {
      key: 'shoulderWidth' as keyof BodyMeasurements,
      label: 'Shoulder Width (cm)',
      placeholder: 'Enter your shoulder width',
      icon: Ruler,
      required: false
    },
    {
      key: 'waist' as keyof BodyMeasurements,
      label: 'Waist (cm)',
      placeholder: 'Enter your waist measurement',
      icon: Ruler,
      required: false
    },
    {
      key: 'hip' as keyof BodyMeasurements,
      label: 'Hip (cm)',
      placeholder: 'Enter your hip measurement',
      icon: Ruler,
      required: false
    }
  ];

  return (
    <div className="space-y-6">
      {measurementFields.map((field, index) => {
        const Icon = field.icon;
        const isActive = activeField === field.key;
        const isFilled = measurements[field.key] && measurements[field.key]! > 0;
        const isHovered = hoveredRegion && regionToField[hoveredRegion] === field.key;
        
        return (
          <motion.div
            key={field.key}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            className={`relative ${isHovered ? 'z-10' : ''}`}
          >
            <label className={getLabelStyling(field.key)}>
              {field.label} {!field.required && '- Optional'}
              {isFilled && (
                <motion.span
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="ml-2 inline-block w-2 h-2 bg-neon-green rounded-full shadow-lg shadow-neon-green/50"
                />
              )}
            </label>
            
            <div className="relative">
              <input
                ref={(el) => { inputRefs.current[field.key] = el; }}
                type="number"
                value={measurements[field.key] || ''}
                onChange={(e) => onMeasurementChange(field.key, Number(e.target.value) || 0)}
                onFocus={() => onFieldFocus(field.key)}
                onBlur={onFieldBlur}
                className={getInputStyling(field.key)}
                placeholder={field.placeholder}
                style={{ paddingLeft: '2.5rem' }}
                data-field={field.key}
              />
              
              <div className="absolute left-3 top-1/2 transform -translate-y-1/2 pointer-events-none">
                <Icon 
                  size={16} 
                  className={
                    isHovered ? "text-neon-pink" :
                    isFilled ? "text-neon-green" :
                    isActive ? "text-neon-blue" :
                    "text-gray-400"
                  }
                />
              </div>
              
              {/* Neon glow effect for hovered/active inputs */}
              {(isHovered || isActive || isFilled) && (
                <motion.div
                  className="absolute inset-0 rounded-xl pointer-events-none"
                  initial={{ opacity: 0 }}
                  animate={{ 
                    opacity: isHovered ? 0.6 : isActive ? 0.4 : 0.2,
                    boxShadow: isHovered 
                      ? "0 0 30px rgba(255, 0, 128, 0.3)" 
                      : isFilled 
                        ? "0 0 20px rgba(0, 255, 136, 0.2)"
                        : "0 0 20px rgba(0, 212, 255, 0.2)"
                  }}
                  transition={{ duration: 0.3 }}
                />
              )}
            </div>
            
          </motion.div>
        );
      })}
      
      {/* Progress indicator */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="mt-8 p-4 bg-dark-card/30 rounded-xl border border-neon-blue/20"
      >
        <div className="flex justify-between text-sm text-gray-400 mb-2 font-oswald">
          <span>Measurement Progress</span>
          <span>
            {[measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length}/3 required
          </span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <motion.div
            className="bg-gradient-to-r from-neon-blue via-neon-green to-neon-pink h-2 rounded-full"
            initial={{ width: 0 }}
            animate={{
              width: `${([measurements.height, measurements.weight, measurements.wingspan].filter(Boolean).length / 3) * 100}%`
            }}
            transition={{ duration: 0.5 }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-2 font-oswald">
          <span>Optional: {[measurements.shoulderWidth, measurements.waist, measurements.hip].filter(Boolean).length}/3</span>
          <span>Total: {Object.values(measurements).filter(Boolean).length}/6</span>
        </div>
      </motion.div>
    </div>
  );
}
