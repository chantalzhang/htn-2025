'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { BodyMeasurements } from '@/types';

// Import SVG assets as React components
import HeadSVG from '@/assets/head.svg';
import TorsoSVG from '@/assets/torso.svg';
import LeftHandSVG from '@/assets/left-hand.svg';
import RightHandSVG from '@/assets/right-hand.svg';
import LeftShoulderSVG from '@/assets/left-shoulder.svg';
import RightShoulderSVG from '@/assets/right-shoulder.svg';
import TrunksSVG from '@/assets/trunks.svg';
import LegsSVG from '@/assets/legs.svg';

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
        return ['head', 'torso', 'legs'];
      case 'wingspan':
        return ['leftHand', 'rightHand'];
      case 'shoulderWidth':
        return ['leftShoulder', 'rightShoulder'];
      case 'waist':
        return ['trunks'];
      case 'hip':
        return ['trunks'];
      default:
        return [];
    }
  };

  const highlightedParts = getHighlightedParts();

  // Helper function to get highlight styles
  const getHighlightStyle = (partName: string) => {
    const isHighlighted = highlightedParts.includes(partName);
    return {
      filter: isHighlighted 
        ? 'drop-shadow(0 0 15px rgba(0, 255, 136, 0.8)) brightness(1.2)' 
        : 'drop-shadow(0 0 10px rgba(0, 212, 255, 0.5))',
      opacity: isHighlighted ? 1 : 0.8,
    };
  };

  return (
    <div className="relative w-full max-w-lg mx-auto">
      {/* Human Figure using SVG assets */}
      <div className="relative flex flex-col items-center">
        <div 
          className="relative"
          style={{ 
            width: '300px', 
            height: '500px',
            filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.3))'
          }}
        >
          {/* Head - positioned at the top */}
          <motion.div
            className="absolute"
            style={{
              top: '0px',
              left: '148px',
              transform: 'translateX(-50%)',
              width: '120px',
              height: '125px',
            }}
            initial={{ scale: 0, y: -20 }}
            animate={{ scale: 0.6, y: 0 }}
            transition={{ duration: 0.8, delay: 0 }}
          >
            <HeadSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('head')
              }}
            />
          </motion.div>

          {/* Left Shoulder - positioned on the left side of torso */}
          <motion.div
            className="absolute"
            style={{
              top: '60px',
              left: '50px',
              width: '200px',
              height: '200px',
            }}
            initial={{ scale: 0, x: -20 }}
            animate={{ scale: 0.5, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <LeftShoulderSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('leftShoulder')
              }}
            />
          </motion.div>

          {/* Right Shoulder - positioned on the right side of torso */}
          <motion.div
            className="absolute"
            style={{
              top: '60px',
              right: '-100px',
              width: '200px',
              height: '200px',
            }}
            initial={{ scale: 0, x: 20 }}
            animate={{ scale: 0.5, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <RightShoulderSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('rightShoulder')
              }}
            />
          </motion.div>

          {/* Torso - positioned in the center */}
          <motion.div
            className="absolute"
            style={{
              top: '50px',
              left: '120px',
              transform: 'translateX(-50%)',
              width: '200px',
              height: '240px',
            }}
            initial={{ scale: 0, y: 20 }}
            animate={{ scale: 0.5, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <TorsoSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('torso')
              }}
            />
          </motion.div>

          {/* Left Hand - extending from left shoulder */}
          <motion.div
            className="absolute"
            style={{
              top: '124px',
              left: '25px',
              width: '160px',
              height: '250px',
            }}
            initial={{ scale: 0, x: -30, rotate: -10 }}
            animate={{ scale: 0.5, x: 0, rotate: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <LeftHandSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('leftHand')
              }}
            />
          </motion.div>

          {/* Right Hand - extending from right shoulder */}
          <motion.div
            className="absolute"
            style={{
              top: '124px',
              right: '-95px',
              width: '160px',
              height: '250px',
            }}
            initial={{ scale: 0, x: 30, rotate: 10 }}
            animate={{ scale: 0.5, x: 0, rotate: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <RightHandSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('rightHand')
              }}
            />
          </motion.div>

          {/* Trunks (waist/hips area) - positioned below torso */}
          <motion.div
            className="absolute"
            style={{
              top: '190px',
              left: '115px',
              transform: 'translateX(-50%)',
              width: '190px',
              height: '170px',
            }}
            initial={{ scale: 0, y: 20 }}
            animate={{ scale: 0.5, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            <TrunksSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('trunks')
              }}
            />
          </motion.div>

          {/* Legs - positioned below trunks */}
          <motion.div
            className="absolute"
            style={{
              top: '230px',
              left: '117px',
              transform: 'translateX(-50%)',
              width: '180px',
              height: '320px',
            }}
            initial={{ scale: 0, y: 30 }}
            animate={{ scale: 0.5, y: 0 }}
            transition={{ duration: 0.8, delay: 1.0 }}
          >
            <LegsSVG 
              style={{
                width: '100%',
                height: '100%',
                ...getHighlightStyle('legs')
              }}
            />
          </motion.div>
        </div>
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
            <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-24 h-4 bg-gradient-to-r from-blue-400 to-pink-400 rounded-full"></div>
            
            {/* Scale Platform */}
            <motion.div
              className="absolute top-0 left-1/2 transform -translate-x-1/2 w-20 h-3 bg-gradient-to-r from-green-400 to-orange-400 rounded-full"
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
            <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-green-400 font-bold text-lg">
              {measurements.weight || 0}kg
            </div>
          </div>
          
          <p className="text-gray-400 font-bold text-sm mt-2">
            Body Weight Scale
          </p>
        </motion.div>
      )}

      {/* Measurement Guide */}
      {activeField && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 bg-gray-800/50 rounded-xl border border-blue-400/30"
        >
          <h4 className="text-blue-400 font-bold text-lg mb-2">
            {activeField.charAt(0).toUpperCase() + activeField.slice(1)} Measurement
          </h4>
          <p className="text-gray-300 font-bold text-sm">
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