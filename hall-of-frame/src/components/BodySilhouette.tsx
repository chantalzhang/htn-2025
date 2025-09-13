'use client';

import { motion } from 'framer-motion';
import { useState } from 'react';
import { BodyMeasurements, BodyPart } from '@/types';
import { bodyParts } from '@/utils/similarity';

interface BodySilhouetteProps {
  measurements: BodyMeasurements;
  onMeasurementChange: (key: keyof BodyMeasurements, value: number) => void;
}

export default function BodySilhouette({ measurements, onMeasurementChange }: BodySilhouetteProps) {
  const [activePart, setActivePart] = useState<string | null>(null);
  const [showInput, setShowInput] = useState(false);

  const handlePartClick = (part: BodyPart) => {
    setActivePart(part.id);
    setShowInput(true);
  };

  const handleInputSubmit = (value: string) => {
    if (activePart) {
      const part = bodyParts.find(p => p.id === activePart);
      if (part && !isNaN(Number(value))) {
        onMeasurementChange(part.measurement, Number(value));
      }
    }
    setShowInput(false);
    setActivePart(null);
  };

  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Body Silhouette SVG */}
      <div className="relative">
        <svg
          viewBox="0 0 120 200"
          className="w-full h-auto max-h-96"
          style={{ filter: 'drop-shadow(0 0 20px rgba(0, 245, 255, 0.3))' }}
        >
          {/* Body outline */}
          <motion.path
            d="M60 10 C50 10, 45 15, 45 25 L45 35 C45 40, 50 45, 60 45 C70 45, 75 40, 75 35 L75 25 C75 15, 70 10, 60 10 Z"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 2 }}
          />
          
          {/* Head */}
          <motion.circle
            cx="60"
            cy="20"
            r="8"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[0])}
            whileHover={{ scale: 1.1, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Torso */}
          <motion.rect
            x="50"
            y="30"
            width="20"
            height="40"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[2])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Arms */}
          <motion.path
            d="M50 35 L35 50 L40 55 L55 40 Z"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[3])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />
          <motion.path
            d="M70 35 L85 50 L80 55 L65 40 Z"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[3])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Waist */}
          <motion.rect
            x="52"
            y="60"
            width="16"
            height="8"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[4])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Hips */}
          <motion.rect
            x="50"
            y="68"
            width="20"
            height="12"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[5])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Legs */}
          <motion.rect
            x="55"
            y="80"
            width="10"
            height="40"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
            className="cursor-pointer hover:stroke-neon-green transition-colors duration-300"
            onClick={() => handlePartClick(bodyParts[6])}
            whileHover={{ scale: 1.05, stroke: "#39ff14" }}
            whileTap={{ scale: 0.95 }}
          />

          {/* Feet */}
          <motion.ellipse
            cx="60"
            cy="125"
            rx="8"
            ry="4"
            fill="none"
            stroke="#00f5ff"
            strokeWidth="2"
          />
        </svg>

        {/* Interactive Labels */}
        {bodyParts.map((part, index) => (
          <motion.div
            key={part.id}
            className="absolute text-xs font-montserrat text-neon-blue opacity-0 hover:opacity-100 transition-opacity duration-300"
            style={{
              left: `${part.position.x}%`,
              top: `${part.position.y}%`,
              transform: 'translate(-50%, -50%)'
            }}
            initial={{ opacity: 0 }}
            whileHover={{ opacity: 1 }}
          >
            {part.name}
          </motion.div>
        ))}
      </div>

      {/* Input Modal */}
      {showInput && activePart && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
          onClick={() => setShowInput(false)}
        >
          <motion.div
            className="card p-6 max-w-sm mx-4"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-xl font-montserrat font-bold mb-4 text-neon-blue">
              {bodyParts.find(p => p.id === activePart)?.name} Measurement
            </h3>
            <p className="text-gray-400 mb-4">
              {bodyParts.find(p => p.id === activePart)?.description}
            </p>
            <div className="space-y-4">
              <input
                type="number"
                placeholder="Enter measurement"
                className="input-field w-full"
                autoFocus
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    handleInputSubmit((e.target as HTMLInputElement).value);
                  }
                }}
              />
              <div className="flex gap-3">
                <button
                  onClick={() => setShowInput(false)}
                  className="btn-secondary flex-1"
                >
                  Cancel
                </button>
                <button
                  onClick={() => {
                    const input = document.querySelector('input[type="number"]') as HTMLInputElement;
                    if (input) handleInputSubmit(input.value);
                  }}
                  className="btn-primary flex-1"
                >
                  Save
                </button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Current Measurements Display */}
      <div className="mt-6 space-y-2">
        <h4 className="text-lg font-montserrat font-bold text-neon-green mb-3">
          Your Measurements
        </h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Height:</span>
            <span className="text-white">{measurements.height || 0} cm</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Weight:</span>
            <span className="text-white">{measurements.weight || 0} kg</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Wingspan:</span>
            <span className="text-white">{measurements.wingspan || 0} cm</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Shoulders:</span>
            <span className="text-white">{measurements.shoulderWidth || 0} cm</span>
          </div>
        </div>
      </div>
    </div>
  );
}
