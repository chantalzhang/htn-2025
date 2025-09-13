'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Share2, RotateCcw, Trophy, Target, BarChart3 } from 'lucide-react';
import { BodyMeasurements, UserResults } from '@/types';
import { findTopAthleteMatches, recommendSports, generateAnalysis } from '@/utils/similarity';
import AthleteCard from '@/components/AthleteCard';
import SportCard from '@/components/SportCard';

export default function ResultsPage() {
  const router = useRouter();
  const [results, setResults] = useState<UserResults | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Get measurements from localStorage
    const storedMeasurements = localStorage.getItem('userMeasurements');
    if (!storedMeasurements) {
      router.push('/input');
      return;
    }

    try {
      const measurements: BodyMeasurements = JSON.parse(storedMeasurements);
      
      // Calculate results
      const topAthletes = findTopAthleteMatches(measurements, 3);
      const topSports = recommendSports(measurements);
      const analysis = generateAnalysis(measurements, topAthletes, topSports);

      setResults({
        topSports,
        topAthletes,
        userMeasurements: measurements,
        analysis
      });
    } catch (error) {
      console.error('Error parsing measurements:', error);
      router.push('/input');
    } finally {
      setLoading(false);
    }
  }, [router]);

  const handleStartOver = () => {
    localStorage.removeItem('userMeasurements');
    router.push('/input');
  };

  const handleShare = () => {
    if (results && results.topAthletes.length > 0) {
      const topMatch = results.topAthletes[0];
      const topSport = results.topSports[0];
      
      const shareText = `I just discovered I'm built like ${topMatch.athlete.name} (${topMatch.athlete.sport})! My top sport match is ${topSport.sport} with a ${Math.round(topSport.score)}% compatibility score. Find your athletic match at Hall of Frame!`;
      
      if (navigator.share) {
        navigator.share({
          title: 'Hall of Frame Results',
          text: shareText,
          url: window.location.origin
        });
      } else {
        navigator.clipboard.writeText(shareText);
        alert('Results copied to clipboard!');
      }
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Trophy className="text-neon-blue" size={48} />
        </motion.div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="min-h-screen bg-dark-bg flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-montserrat text-white mb-4">No results found</h1>
          <button onClick={() => router.push('/input')} className="btn-primary">
            Start Over
          </button>
        </div>
      </div>
    );
  }

  const topMatch = results.topAthletes[0];
  const topSport = results.topSports[0];

  return (
    <div className="min-h-screen bg-dark-bg py-8 px-4">
      <div className="max-w-7xl mx-auto">
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
          <div className="flex gap-3">
            <button
              onClick={handleShare}
              className="flex items-center gap-2 btn-secondary"
            >
              <Share2 size={16} />
              Share
            </button>
            <button
              onClick={handleStartOver}
              className="flex items-center gap-2 btn-primary"
            >
              <RotateCcw size={16} />
              Try Again
            </button>
          </div>
        </motion.div>

        {/* Hero Results */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <motion.h1
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="text-4xl md:text-6xl font-bebas gradient-text mb-4"
          >
            YOU'RE BUILT LIKE
          </motion.h1>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-3xl md:text-5xl font-bebas text-neon-blue mb-2"
          >
            {topMatch.athlete.name}
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl md:text-2xl text-neon-green mb-6"
          >
            {topMatch.athlete.sport} • {Math.round(topMatch.similarityScore)}% Match
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-lg text-gray-400 max-w-3xl mx-auto leading-relaxed"
          >
            {results.analysis}
          </motion.p>
        </motion.div>

        {/* Top Sports Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-8">
            <Target className="text-neon-gold" size={32} />
            <h2 className="text-3xl font-bebas text-neon-gold">
              YOUR TOP SPORT MATCHES
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {results.topSports.map((sport, index) => (
              <SportCard key={sport.sport} sport={sport} index={index} />
            ))}
          </div>
        </motion.div>

        {/* Top Athletes Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-8">
            <Trophy className="text-neon-blue" size={32} />
            <h2 className="text-3xl font-bebas text-neon-blue">
              YOUR ATHLETE MATCHES
            </h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {results.topAthletes.map((match, index) => (
              <AthleteCard key={match.athlete.id} match={match} index={index} />
            ))}
          </div>
        </motion.div>

        {/* Your Measurements Summary */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.4 }}
          className="card p-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="text-neon-green" size={32} />
            <h2 className="text-3xl font-bebas text-neon-green">
              YOUR MEASUREMENTS
            </h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bebas text-neon-blue mb-2">
                {results.userMeasurements.height} cm
              </div>
              <div className="text-gray-400">Height</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bebas text-neon-green mb-2">
                {results.userMeasurements.weight} kg
              </div>
              <div className="text-gray-400">Weight</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bebas text-neon-gold mb-2">
                {results.userMeasurements.wingspan} cm
              </div>
              <div className="text-gray-400">Wingspan</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bebas text-neon-blue mb-2">
                {results.userMeasurements.shoulderWidth || 'N/A'} cm
              </div>
              <div className="text-gray-400">Shoulder Width</div>
            </div>
          </div>
        </motion.div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.6 }}
          className="text-center mt-12"
        >
          <h3 className="text-2xl font-montserrat font-bold text-white mb-4">
            Ready to start your athletic journey?
          </h3>
          <p className="text-gray-400 mb-6">
            Use these insights to guide your training and discover your potential in {topSport.sport}!
          </p>
          <div className="flex gap-4 justify-center">
            <button
              onClick={handleStartOver}
              className="btn-primary"
            >
              Try Different Measurements
            </button>
            <button
              onClick={handleShare}
              className="btn-secondary"
            >
              Share Your Results
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
