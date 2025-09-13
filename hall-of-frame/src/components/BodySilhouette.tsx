'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { BodyMeasurements } from '@/types';

interface BodySilhouetteProps {
  measurements: BodyMeasurements;
  onMeasurementChange: (key: keyof BodyMeasurements, value: number) => void;
  activeField?: keyof BodyMeasurements | null;
}

export default function BodySilhouette({ measurements, onMeasurementChange, activeField }: BodySilhouetteProps) {
  const [hoveredPart, setHoveredPart] = useState<string | null>(null);

  // Determine which body parts should be highlighted based on active field
  const getHighlightedParts = () => {
    if (!activeField) return [];
    
    switch (activeField) {
      case 'height':
        return ['head', 'torso', 'legs', 'feet'];
      case 'wingspan':
        return ['leftArm', 'rightArm'];
      case 'shoulderWidth':
        return ['leftShoulder', 'rightShoulder'];
      case 'waist':
        return ['waist'];
      case 'hip':
        return ['hips'];
      default:
        return [];
    }
  };

  const highlightedParts = getHighlightedParts();

  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Human Figure SVG */}
      <div className="relative">
        <svg
          viewBox="0 0 200 300"
          className="w-full h-auto max-h-96"
          style={{ filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.3))' }}
        >
          {/* Head */}
          <motion.circle
            cx="100"
            cy="30"
            r="20"
            fill={highlightedParts.includes('head') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('head') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('head') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1 }}
          />
          
          {/* Torso */}
          <motion.rect
            x="70"
            y="50"
            width="60"
            height="80"
            rx="10"
            fill={highlightedParts.includes('torso') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('torso') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('torso') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
          />
          
          {/* Left Arm */}
          <motion.path
            d="M70 70 L40 100 L35 120 L45 125 L55 110 Z"
            fill={highlightedParts.includes('leftArm') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('leftArm') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('leftArm') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
          />
          
          {/* Right Arm */}
          <motion.path
            d="M130 70 L160 100 L165 120 L155 125 L145 110 Z"
            fill={highlightedParts.includes('rightArm') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('rightArm') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('rightArm') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 0.4 }}
          />
          
          {/* Left Shoulder */}
          <motion.circle
            cx="70"
            cy="70"
            r="8"
            fill={highlightedParts.includes('leftShoulder') ? "rgba(0, 255, 136, 0.5)" : "none"}
            stroke={highlightedParts.includes('leftShoulder') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('leftShoulder') ? "4" : "2"}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          />
          
          {/* Right Shoulder */}
          <motion.circle
            cx="130"
            cy="70"
            r="8"
            fill={highlightedParts.includes('rightShoulder') ? "rgba(0, 255, 136, 0.5)" : "none"}
            stroke={highlightedParts.includes('rightShoulder') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('rightShoulder') ? "4" : "2"}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          />
          
          {/* Waist */}
          <motion.rect
            x="75"
            y="90"
            width="50"
            height="15"
            rx="7"
            fill={highlightedParts.includes('waist') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('waist') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('waist') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
          />
          
          {/* Hips */}
          <motion.rect
            x="70"
            y="105"
            width="60"
            height="20"
            rx="10"
            fill={highlightedParts.includes('hips') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('hips') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('hips') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1 }}
          />
          
          {/* Left Leg */}
          <motion.rect
            x="80"
            y="125"
            width="15"
            height="60"
            rx="7"
            fill={highlightedParts.includes('legs') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('legs') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('legs') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.2 }}
          />
          
          {/* Right Leg */}
          <motion.rect
            x="105"
            y="125"
            width="15"
            height="60"
            rx="7"
            fill={highlightedParts.includes('legs') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('legs') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('legs') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.2 }}
          />
          
          {/* Left Foot */}
          <motion.ellipse
            cx="87"
            cy="190"
            rx="12"
            ry="6"
            fill={highlightedParts.includes('feet') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('feet') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('feet') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.4 }}
          />
          
          {/* Right Foot */}
          <motion.ellipse
            cx="113"
            cy="190"
            rx="12"
            ry="6"
            fill={highlightedParts.includes('feet') ? "rgba(0, 255, 136, 0.3)" : "none"}
            stroke={highlightedParts.includes('feet') ? "#00ff88" : "#00d4ff"}
            strokeWidth={highlightedParts.includes('feet') ? "4" : "3"}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1, delay: 1.4 }}
          />
        </svg>
      </div>

      {/* Weight Scale (Abstract) */}
      {activeField === 'weight' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 text-center"
        >
          <div className="relative w-32 h-16 mx-auto">
            {/* Scale Base */}
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-24 h-4 bg-gradient-to-r from-neon-blue to-neon-pink rounded-full"></div>
            
            {/* Scale Platform */}
            <motion.div
              className="absolute top-0 left-1/2 transform -translate-x-1/2 w-20 h-3 bg-gradient-to-r from-neon-green to-neon-orange rounded-full"
              animate={{ 
                y: [0, -5, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ 
                duration: 2,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            ></motion.div>
            
            {/* Weight Value Display */}
            <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-neon-green font-oswald font-bold text-lg">
              {measurements.weight || 0}kg
            </div>
          </div>
          
          <p className="text-text-secondary font-oswald text-sm mt-2">
            Body Weight Scale
          </p>
        </motion.div>
      )}

      {/* Measurement Guide */}
      {activeField && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-dark-card/50 rounded-xl border border-neon-blue/30"
        >
          <h4 className="text-neon-blue font-oswald font-bold text-lg mb-2">
            {activeField.charAt(0).toUpperCase() + activeField.slice(1)} Measurement
          </h4>
          <p className="text-text-secondary font-oswald text-sm">
            {activeField === 'height' && "Measure from the top of your head to the bottom of your feet"}
            {activeField === 'weight' && "Step on a scale to measure your body weight"}
            {activeField === 'wingspan' && "Stretch your arms out horizontally and measure fingertip to fingertip"}
            {activeField === 'shoulderWidth' && "Measure across your shoulders from the outside edge of one shoulder to the other"}
            {activeField === 'waist' && "Measure around the narrowest part of your torso, usually just above your belly button"}
            {activeField === 'hip' && "Measure around the widest part of your hips, usually at the level of your hip bones"}
          </p>
        </motion.div>
      )}
    </div>
  );
}