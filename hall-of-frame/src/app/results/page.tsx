'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, Share2, RotateCcw, Trophy, Target, BarChart3, Zap, Activity, Shield, Gauge } from 'lucide-react';
import { BodyMeasurements, UserResults, AthleticStats } from '@/types';
import { findTopAthleteMatches, recommendSports, generateAnalysis, calculateAthleticStats } from '@/utils/similarity';
import AthleteCard from '@/components/AthleteCard';
import SportCard from '@/components/SportCard';
import SpiderChart from '@/components/SpiderChart';

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
      const athleticStats = calculateAthleticStats(measurements, topAthletes);

      setResults({
        topSports,
        topAthletes,
        userMeasurements: measurements,
        analysis,
        athleticStats
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
    <div className="min-h-screen py-8 px-4 relative overflow-hidden">
      {/* Bold Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary-blue/5 via-transparent to-neon-pink/5"></div>
        <div className="absolute top-1/4 right-0 w-96 h-96 bg-gradient-radial from-neon-blue/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute bottom-1/4 left-0 w-80 h-80 bg-gradient-radial from-neon-pink/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-neon-green/10 to-transparent rounded-full blur-2xl"></div>
      </div>
      
      <div className="max-w-7xl mx-auto relative z-10">
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
            <span className="font-oswald">Back</span>
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
            className="text-5xl md:text-7xl font-oswald gradient-text mb-6 font-black"
            style={{ 
              textShadow: '0 0 40px rgba(0, 212, 255, 0.5), 0 0 80px rgba(255, 0, 128, 0.3)',
              letterSpacing: '0.05em'
            }}
          >
            YOU'RE BUILT FOR
          </motion.h1>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-4xl md:text-6xl font-oswald text-neon-blue mb-3 font-black"
            style={{ textShadow: '0 0 30px rgba(0, 212, 255, 0.6)' }}
          >
            {topSport.sport}
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-2xl md:text-3xl text-neon-green mb-8 font-bold"
            style={{ textShadow: '0 0 20px rgba(0, 255, 136, 0.5)' }}
          >
            {Math.round(topSport.score)}% Match
          </motion.div>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-xl text-text-secondary max-w-4xl mx-auto leading-relaxed font-oswald font-medium"
          >
            {results.analysis}
          </motion.p>
        </motion.div>

        {/* Athlete Match Section */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="text-center mb-12"
        >
          <div className="card-blue p-6 max-w-2xl mx-auto">
            <h3 className="text-2xl font-oswald font-black text-neon-blue mb-4">
              Most Similar Athlete
            </h3>
            <div className="text-xl font-oswald text-white mb-2">
              {topMatch.athlete.name}
            </div>
            <div className="text-lg text-neon-green font-oswald font-bold">
              {topMatch.athlete.sport} • {Math.round(topMatch.similarityScore)}% Match
            </div>
          </div>
        </motion.div>

        {/* Athletic Stats Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-8">
            <Gauge className="text-neon-orange" size={32} />
            <h2 className="text-4xl font-oswald text-neon-orange font-black">
              YOUR ATHLETIC PROFILE
            </h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Spider Chart */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 1 }}
              className="card-blue p-8 text-center group hover:shadow-2xl hover:shadow-blue-500/20 transition-all duration-500"
            >
              <h3 className="text-2xl font-oswald font-black text-neon-blue mb-6 group-hover:text-white transition-colors duration-300">
                Athletic Radar Chart
              </h3>
              <SpiderChart stats={results.athleticStats} className="mx-auto" />
              <p className="text-text-secondary font-oswald mt-4 group-hover:text-white transition-colors duration-300">
                Hover over the data points to see detailed stats
              </p>
            </motion.div>

            {/* Individual Stats */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, delay: 1.2 }}
              className="card-pink p-8"
            >
              <h3 className="text-2xl font-oswald font-black text-neon-pink mb-6">
                Detailed Stats
              </h3>
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(results.athleticStats).map(([stat, value], index) => {
                  const statIcons = {
                    strength: Shield,
                    agility: Zap,
                    endurance: Activity,
                    power: Trophy,
                    speed: Target,
                    flexibility: BarChart3,
                    coordination: Gauge,
                    balance: Shield
                  };
                  const IconComponent = statIcons[stat as keyof AthleticStats];
                  const statName = stat.charAt(0).toUpperCase() + stat.slice(1);
                  
                  return (
                    <motion.div
                      key={stat}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.5, delay: 1.4 + index * 0.1 }}
                      whileHover={{ 
                        scale: 1.05, 
                        y: -5,
                        boxShadow: "0 10px 25px rgba(236, 72, 153, 0.3)"
                      }}
                      className="bg-dark-card/50 rounded-xl p-4 border border-gray-700 hover:border-neon-pink transition-all duration-300 cursor-pointer"
                    >
                      <div className="flex items-center gap-3 mb-2">
                        <IconComponent className="text-neon-pink" size={20} />
                        <span className="font-oswald font-bold text-text-secondary text-sm">
                          {statName}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-700 rounded-full h-2">
                          <motion.div
                            className="bg-gradient-to-r from-neon-pink to-neon-blue h-2 rounded-full"
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.round(value)}%` }}
                            transition={{ duration: 1, delay: 1.6 + index * 0.1 }}
                          />
                        </div>
                        <span className="font-oswald font-black text-neon-pink text-lg">
                          {Math.round(value)}
                        </span>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          </div>
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
            <h2 className="text-4xl font-oswald text-neon-gold font-black">
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
            <h2 className="text-4xl font-oswald text-neon-blue font-black">
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
          className="card-green p-10"
        >
          <div className="flex items-center gap-3 mb-6">
            <BarChart3 className="text-neon-green" size={32} />
            <h2 className="text-4xl font-oswald text-neon-green font-black">
              YOUR MEASUREMENTS
            </h2>
          </div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-oswald text-neon-blue mb-2">
                {results.userMeasurements.height} cm
              </div>
              <div className="text-gray-400 font-oswald">Height</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-oswald text-neon-green mb-2">
                {results.userMeasurements.weight} kg
              </div>
              <div className="text-gray-400 font-oswald">Weight</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-oswald text-neon-gold mb-2">
                {results.userMeasurements.wingspan} cm
              </div>
              <div className="text-gray-400 font-oswald">Wingspan</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-oswald text-neon-blue mb-2">
                {results.userMeasurements.shoulderWidth || 'N/A'} cm
              </div>
              <div className="text-gray-400 font-oswald">Shoulder Width</div>
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
          <h3 className="text-3xl font-oswald font-black text-white mb-6">
            Ready to start your athletic journey?
          </h3>
          <p className="text-text-secondary text-xl mb-8 font-oswald font-medium">
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
