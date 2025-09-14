'use client';

import { motion } from 'framer-motion';
import { BodyMeasurements } from '@/types';

interface BodyMapProps {
  measurements: BodyMeasurements;
  activeField: keyof BodyMeasurements | null;
  hoveredRegion: string | null;
  onRegionHover: (region: string | null) => void;
  onRegionClick: (field: keyof BodyMeasurements) => void;
}

export default function BodyMap({ 
  measurements, 
  activeField, 
  hoveredRegion, 
  onRegionHover, 
  onRegionClick 
}: BodyMapProps) {
  
  // Map body regions to measurement fields
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

  // Determine which regions should be highlighted
  const getHighlightedRegions = () => {
    const regions: string[] = [];
    
    if (activeField) {
      switch (activeField) {
        case 'height':
          regions.push('head', 'torso', 'legs', 'feet');
          break;
        case 'wingspan':
          regions.push('leftArm', 'rightArm');
          break;
        case 'shoulderWidth':
          regions.push('leftShoulder', 'rightShoulder');
          break;
        case 'waist':
          regions.push('waist');
          break;
        case 'hip':
          regions.push('hips');
          break;
        case 'weight':
          regions.push('weight');
          break;
      }
    }
    
    if (hoveredRegion) {
      regions.push(hoveredRegion);
    }
    
    return regions;
  };

  // Check if a region has a filled measurement
  const isRegionFilled = (region: string) => {
    const field = regionToField[region];
    return field && measurements[field] && measurements[field]! > 0;
  };

  // Get region color based on state
  const getRegionColor = (region: string) => {
    const isHighlighted = getHighlightedRegions().includes(region);
    const isFilled = isRegionFilled(region);
    const isHovered = hoveredRegion === region;
    
    if (isHovered) {
      return {
        fill: "rgba(255, 0, 128, 0.4)",
        stroke: "#ff0080",
        strokeWidth: "5",
        filter: "drop-shadow(0 0 15px #ff0080)"
      };
    }
    
    if (isFilled) {
      return {
        fill: "rgba(0, 255, 136, 0.3)",
        stroke: "#00ff88",
        strokeWidth: "4",
        filter: "drop-shadow(0 0 10px #00ff88)"
      };
    }
    
    if (isHighlighted) {
      return {
        fill: "rgba(0, 212, 255, 0.3)",
        stroke: "#00d4ff",
        strokeWidth: "4",
        filter: "drop-shadow(0 0 10px #00d4ff)"
      };
    }
    
    return {
      fill: "none",
      stroke: "#4a5568",
      strokeWidth: "2",
      filter: "none"
    };
  };

  const handleRegionInteraction = (region: string, action: 'hover' | 'click') => {
    const field = regionToField[region];
    if (field) {
      if (action === 'hover') {
        onRegionHover(region);
      } else {
        onRegionClick(field);
      }
    }
  };

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative p-12">
        <svg
          width="400"
          height="600"
          viewBox="0 0 400 600"
          className="mx-auto"
        >
          {/* Head - Clean oval silhouette */}
          <motion.path
            id="head"
            d="M 200 30 C 220 30 235 45 235 70 C 235 95 220 110 200 110 C 180 110 165 95 165 70 C 165 45 180 30 200 30 Z"
            fill="none"
            stroke={getRegionColor('head').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('head').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('head')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['head'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('head').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
          
          {/* Shoulders - Clean shoulder line */}
          <motion.path
            id="shoulders"
            d="M 140 120 C 150 115 170 110 200 110 C 230 110 250 115 260 120 C 270 125 275 135 270 140 C 260 145 230 150 200 150 C 170 150 140 145 130 140 C 125 135 130 125 140 120 Z"
            fill="none"
            stroke={getRegionColor('leftShoulder').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('leftShoulder').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('leftShoulder')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['leftShoulder'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('leftShoulder').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.2, ease: "easeInOut" }}
          />
          
          {/* Torso - Clean torso outline */}
          <motion.path
            id="torso"
            d="M 175 150 C 170 155 168 170 170 190 C 172 210 175 225 180 240 C 185 250 190 255 200 255 C 210 255 215 250 220 240 C 225 225 228 210 230 190 C 232 170 230 155 225 150 C 215 145 185 145 175 150 Z"
            fill="none"
            stroke={getRegionColor('torso').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('torso').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('torso')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['torso'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('torso').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.3, ease: "easeInOut" }}
          />
          
          {/* Arms - Clean arm outlines */}
          <motion.path
            id="left-arm"
            d="M 140 140 C 120 145 105 160 100 180 C 95 200 100 220 110 235 C 120 245 135 250 145 245 C 155 240 160 225 158 210 C 156 195 152 180 148 165 C 145 155 142 150 140 140 Z"
            fill="none"
            stroke={getRegionColor('arms').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('arms').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('arms')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['arms'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('arms').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.4, ease: "easeInOut" }}
          />
          
          <motion.path
            id="right-arm"
            d="M 260 140 C 280 145 295 160 300 180 C 305 200 300 220 290 235 C 280 245 265 250 255 245 C 245 240 240 225 242 210 C 244 195 248 180 252 165 C 255 155 258 150 260 140 Z"
            fill="none"
            stroke={getRegionColor('arms').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('arms').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('arms')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['arms'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('arms').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.4, ease: "easeInOut" }}
          />
          
          {/* Waist - Clean waist outline */}
          <motion.path
            id="waist"
            d="M 185 255 C 180 260 178 270 180 280 C 182 290 188 295 200 295 C 212 295 218 290 220 280 C 222 270 220 260 215 255 C 210 250 190 250 185 255 Z"
            fill="none"
            stroke={getRegionColor('waist').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('waist').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('waist')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['waist'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('waist').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.5, ease: "easeInOut" }}
          />
          
          {/* Hips - Clean hip outline */}
          <motion.path
            id="hips"
            d="M 160 295 C 150 300 145 315 148 330 C 151 345 160 355 175 360 C 190 365 210 365 225 360 C 240 355 249 345 252 330 C 255 315 250 300 240 295 C 225 290 175 290 160 295 Z"
            fill="none"
            stroke={getRegionColor('hips').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('hips').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('hips')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['hips'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('hips').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.6, ease: "easeInOut" }}
          />
          
          {/* Legs - Clean leg outlines */}
          <motion.path
            id="legs"
            d="M 175 360 C 170 370 168 390 170 420 C 172 450 175 480 180 500 C 185 520 190 530 195 535 M 225 360 C 230 370 232 390 230 420 C 228 450 225 480 220 500 C 215 520 210 530 205 535"
            fill="none"
            stroke={getRegionColor('legs').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('legs').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('legs')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['legs'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('legs').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.7, ease: "easeInOut" }}
          />
          
          {/* Feet - Clean foot outlines */}
          <motion.path
            id="feet"
            d="M 170 535 C 160 540 155 550 160 560 C 165 570 180 575 195 570 C 205 565 210 555 205 545 M 230 535 C 240 540 245 550 240 560 C 235 570 220 575 205 570 C 195 565 190 555 195 545"
            fill="none"
            stroke={getRegionColor('feet').stroke}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="transition-all duration-300 cursor-pointer hover:drop-shadow-lg"
            style={{
              filter: `drop-shadow(0 0 8px ${getRegionColor('feet').stroke}40)`
            }}
            onMouseEnter={() => onRegionHover('feet')}
            onMouseLeave={() => onRegionHover(null)}
            onClick={() => onRegionClick(regionToField['feet'])}
            whileHover={{ 
              strokeWidth: 4,
              filter: `drop-shadow(0 0 12px ${getRegionColor('feet').stroke}80)`
            }}
            whileTap={{ scale: 0.98 }}
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.8, delay: 0.8, ease: "easeInOut" }}
          />

          {/* Weight Scale - Clean minimal scale */}
          {(activeField === 'weight' || hoveredRegion === 'weight') && (
            <motion.g
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              {/* Scale Base */}
              <motion.path
                id="weight-scale"
                d="M 170 580 C 160 585 160 595 170 600 C 185 605 215 605 230 600 C 240 595 240 585 230 580 C 215 575 185 575 170 580 Z"
                fill="none"
                stroke={getRegionColor('weight').stroke}
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="cursor-pointer hover:drop-shadow-lg"
                style={{
                  filter: `drop-shadow(0 0 8px ${getRegionColor('weight').stroke}40)`
                }}
                onMouseEnter={() => onRegionHover('weight')}
                onMouseLeave={() => onRegionHover(null)}
                onClick={() => onRegionClick(regionToField['weight'])}
                whileHover={{ 
                  strokeWidth: 4,
                  filter: `drop-shadow(0 0 12px ${getRegionColor('weight').stroke}80)`
                }}
                whileTap={{ scale: 0.98 }}
                animate={{ 
                  y: [0, -2, 0],
                  opacity: [0.8, 1, 0.8]
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
              />
              
              {/* Weight Display */}
              <text
                x="200"
                y="620"
                textAnchor="middle"
                className="fill-neon-green font-oswald font-bold text-sm"
              >
                {measurements.weight || 0}kg
              </text>
            </motion.g>
          )}

          {/* Gradient Definitions */}
          <defs>
            <linearGradient id="scaleGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00d4ff" />
              <stop offset="50%" stopColor="#ff00ff" />
              <stop offset="100%" stopColor="#00ff88" />
            </linearGradient>
            <linearGradient id="displayGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00ff88" />
              <stop offset="100%" stopColor="#00d4ff" />
            </linearGradient>
          </defs>
        </svg>
      </div>

      {/* Hover Tooltip */}
      {hoveredRegion && (
        <motion.div
          initial={{ opacity: 0, y: 10, scale: 0.9 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 10, scale: 0.9 }}
          className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-20"
        >
          <div className="bg-dark-card/95 backdrop-blur-md px-4 py-3 rounded-xl border border-neon-pink/50 shadow-lg shadow-neon-pink/25">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <motion.div
                  className="w-2 h-2 bg-neon-pink rounded-full"
                  animate={{ scale: [1, 1.3, 1] }}
                  transition={{ duration: 1, repeat: Infinity }}
                />
                <span className="text-neon-pink font-oswald font-bold text-sm">
                  Fill {regionToField[hoveredRegion]?.replace(/([A-Z])/g, ' $1').toLowerCase()} measurement
                </span>
              </div>
              <motion.div
                animate={{ x: [0, 5, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-neon-pink"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8.59 16.59L13.17 12L8.59 7.41L10 6l6 6-6 6-1.41-1.41z"/>
                </svg>
              </motion.div>
            </div>
            <div className="text-center mt-1">
              <span className="text-xs text-text-secondary font-oswald">Click to focus input</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
