'use client';

import { motion } from 'framer-motion';
import { SportRecommendation } from '@/types';
import { Target, TrendingUp, Star } from 'lucide-react';

interface SportCardProps {
  sport: SportRecommendation;
  index: number;
}

export default function SportCard({ sport, index }: SportCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.2 }}
      whileHover={{ scale: 1.02, y: -5 }}
      className="card card-hover p-6 relative overflow-hidden"
    >

      {/* Sport Icon */}
      <div className="text-6xl mb-4 text-center">
        {sport.icon}
      </div>

      {/* Sport Info */}
      <div className="text-center mb-4">
        <h3 className="text-3xl font-montserrat font-bold text-white mb-3">
          {sport.sport}
        </h3>
        <p className="text-gray-400 text-base leading-relaxed">
          {sport.description}
        </p>
      </div>

      {/* Why Match Section */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-3">
          <Target className="text-neon-green" size={18} />
          <span className="text-base font-montserrat font-medium text-neon-green">
            Why You Match
          </span>
        </div>
        <div className="space-y-2">
          {sport.whyMatch.map((reason, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: index * 0.2 + idx * 0.1 }}
              className="flex items-start gap-2"
            >
              <Star className="text-neon-gold mt-0.5 flex-shrink-0" size={14} />
              <span className="text-sm text-gray-300">{reason}</span>
            </motion.div>
          ))}
        </div>
      </div>

    </motion.div>
  );
}
