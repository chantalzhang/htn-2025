'use client';

import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';

interface SpiderChartProps {
  stats: {
    strength: number;
    agility: number;
    endurance: number;
    power: number;
    speed: number;
    flexibility: number;
    coordination: number;
    balance: number;
  };
  className?: string;
}

export default function SpiderChart({ stats, className = '' }: SpiderChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredStat, setHoveredStat] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  const handleMouseMove = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    setMousePos({ x: event.clientX, y: event.clientY });

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = 120;
    const statCount = 8;
    const angleStep = (2 * Math.PI) / statCount;

    // Check which stat is being hovered
    let hoveredIndex = null;
    for (let i = 0; i < statCount; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const statValue = Object.values(stats)[i];
      const valueRadius = (radius * statValue) / 100;
      const statX = centerX + Math.cos(angle) * valueRadius;
      const statY = centerY + Math.sin(angle) * valueRadius;
      
      const distance = Math.sqrt((x - statX) ** 2 + (y - statY) ** 2);
      if (distance <= 15) { // Hover radius
        hoveredIndex = i;
        break;
      }
    }

    setHoveredStat(hoveredIndex);
  };

  const handleMouseLeave = () => {
    setHoveredStat(null);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const size = 400;
    canvas.width = size;
    canvas.height = size;

    // Clear canvas
    ctx.clearRect(0, 0, size, size);

    // Center point
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = 120;

    // Number of stats
    const statCount = 8;
    const angleStep = (2 * Math.PI) / statCount;

    // Stat labels
    const statLabels = [
      'Strength', 'Agility', 'Endurance', 'Power',
      'Speed', 'Flexibility', 'Coordination', 'Balance'
    ];

    // Draw grid circles
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 5; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (radius * i) / 5, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < statCount; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(x, y);
      ctx.stroke();
    }

    // Draw stat labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '12px Oswald';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    for (let i = 0; i < statCount; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const labelRadius = radius + 40;
      const x = centerX + Math.cos(angle) * labelRadius;
      const y = centerY + Math.sin(angle) * labelRadius;
      
      ctx.fillText(statLabels[i], x, y);
    }


    // Draw the spider web
    const statValues = Object.values(stats);
    ctx.strokeStyle = '#00d4ff';
    ctx.fillStyle = 'rgba(0, 212, 255, 0.2)';
    ctx.lineWidth = 3;

    ctx.beginPath();
    for (let i = 0; i < statCount; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const valueRadius = (radius * statValues[i]) / 100;
      const x = centerX + Math.cos(angle) * valueRadius;
      const y = centerY + Math.sin(angle) * valueRadius;
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Draw data points with hover effects
    for (let i = 0; i < statCount; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const statValue = Object.values(stats)[i];
      const valueRadius = (radius * statValue) / 100;
      const x = centerX + Math.cos(angle) * valueRadius;
      const y = centerY + Math.sin(angle) * valueRadius;
      
      // Hover effect
      if (hoveredStat === i) {
        // Glow effect
        ctx.shadowColor = '#00d4ff';
        ctx.shadowBlur = 20;
        ctx.fillStyle = '#ff0080';
      } else {
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#00d4ff';
      }
      
      ctx.beginPath();
      ctx.arc(x, y, hoveredStat === i ? 8 : 4, 0, 2 * Math.PI);
      ctx.fill();
      
      // Reset shadow
      ctx.shadowBlur = 0;
    }
  }, [stats, hoveredStat]);

  const statLabels = [
    'Strength', 'Agility', 'Endurance', 'Power',
    'Speed', 'Flexibility', 'Coordination', 'Balance'
  ];

  return (
    <div className={`relative flex justify-center ${className}`}>
      <canvas
        ref={canvasRef}
        className="cursor-pointer"
        style={{ maxWidth: '400px', maxHeight: '400px' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      
      {/* Tooltip */}
      {hoveredStat !== null && (
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          className="absolute pointer-events-none z-10 bg-dark-card border border-neon-blue rounded-lg p-3 shadow-2xl"
          style={{
            left: mousePos.x - 200,
            top: mousePos.y - 180,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="text-center">
            <div className="text-neon-blue font-oswald font-black text-lg">
              {statLabels[hoveredStat]}
            </div>
            <div className="text-white font-oswald text-2xl font-bold">
              {Math.round(Object.values(stats)[hoveredStat])}
            </div>
            <div className="text-text-secondary font-oswald text-sm">
              out of 100
            </div>
          </div>
          
        </motion.div>
      )}
    </div>
  );
}
