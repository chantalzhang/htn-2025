'use client';

import { motion } from 'framer-motion';
import { AthleteMatch } from '@/types';
import { Trophy, Target, Award } from 'lucide-react';

interface AthleteCardProps {
  match: AthleteMatch;
  index: number;
}

export default function AthleteCard({ match, index }: AthleteCardProps) {
  const { athlete, similarityScore, matchingTraits } = match;

  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.2 }}
      whileHover={{ scale: 1.02, y: -5 }}
      className="card card-hover p-6 relative overflow-hidden"
    >
      {/* Similarity Score Badge */}
      <div className="absolute top-4 right-4">
        <div className="bg-gradient-to-r from-neon-blue to-neon-green text-dark-bg px-3 py-1 rounded-full text-sm font-bold">
          {Math.round(similarityScore)}% Match
        </div>
      </div>

      {/* Athlete Image Placeholder */}
      <div className="w-20 h-20 bg-gradient-to-br from-neon-blue/20 to-neon-green/20 rounded-full mx-auto mb-4 flex items-center justify-center">
        <Trophy className="text-neon-blue" size={32} />
      </div>

      {/* Athlete Info */}
      <div className="text-center mb-4">
        <h3 className="text-xl font-montserrat font-bold text-white mb-1">
          {athlete.name}
        </h3>
        <div className="flex items-center justify-center gap-2 text-neon-green mb-2">
          <span className="text-sm font-montserrat">{athlete.sport}</span>
          {athlete.position && (
            <>
              <span className="text-gray-400">•</span>
              <span className="text-sm text-gray-400">{athlete.position}</span>
            </>
          )}
        </div>
        <p className="text-sm text-gray-400">{athlete.description}</p>
      </div>

      {/* Measurements Comparison */}
      <div className="space-y-2 mb-4">
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Height:</span>
          <span className="text-white">{athlete.height} cm</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Weight:</span>
          <span className="text-white">{athlete.weight} kg</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-gray-400">Wingspan:</span>
          <span className="text-white">{athlete.wingspan} cm</span>
        </div>
      </div>

      {/* Matching Traits */}
      {matchingTraits.length > 0 && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className="text-neon-gold" size={16} />
            <span className="text-sm font-montserrat font-medium text-neon-gold">
              Matching Traits
            </span>
          </div>
          <div className="flex flex-wrap gap-1">
            {matchingTraits.map((trait, idx) => (
              <span
                key={idx}
                className="px-2 py-1 bg-neon-gold/20 text-neon-gold text-xs rounded-full"
              >
                {trait}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Achievements */}
      {athlete.achievements && athlete.achievements.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Award className="text-neon-blue" size={16} />
            <span className="text-sm font-montserrat font-medium text-neon-blue">
              Key Achievements
            </span>
          </div>
          <div className="space-y-1">
            {athlete.achievements.slice(0, 2).map((achievement, idx) => (
              <div key={idx} className="text-xs text-gray-400">
                • {achievement}
              </div>
            ))}
            {athlete.achievements.length > 2 && (
              <div className="text-xs text-neon-blue">
                +{athlete.achievements.length - 2} more...
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  );
}
