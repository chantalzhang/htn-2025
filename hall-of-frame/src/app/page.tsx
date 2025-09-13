'use client';

import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { Trophy, Target, Zap, Users } from 'lucide-react';

export default function LandingPage() {
  const router = useRouter();

  const handleGetStarted = () => {
    router.push('/input');
  };

  return (
    <div className="min-h-screen bg-dark-bg relative overflow-hidden">
      {/* Hero Section */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl mx-auto"
        >
          {/* Main Title */}
          <motion.h1
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
            className="text-6xl md:text-8xl font-bebas mb-6 gradient-text"
          >
            HALL OF FRAME
          </motion.h1>

          {/* Tagline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl md:text-2xl font-montserrat text-gray-300 mb-8 max-w-2xl mx-auto"
          >
            Does your frame belong in the Hall of Fame?
          </motion.p>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-lg md:text-xl text-gray-400 mb-12 max-w-3xl mx-auto leading-relaxed"
          >
            Discover which sport you're best suited for by comparing your body measurements 
            against elite athletes from around the world. Find your perfect athletic match!
          </motion.p>

          {/* CTA Button */}
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleGetStarted}
            className="btn-primary text-2xl px-12 py-4 neon-glow-blue hover:neon-glow-green transition-all duration-300"
          >
            <Trophy className="inline-block mr-3" size={32} />
            Find Your Sport
          </motion.button>
        </motion.div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1 }}
          className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl mx-auto"
        >
          <motion.div
            whileHover={{ scale: 1.05, y: -10 }}
            className="card card-hover text-center p-8"
          >
            <Target className="mx-auto mb-4 text-neon-blue" size={48} />
            <h3 className="text-xl font-montserrat font-bold mb-3 text-neon-blue">
              Precise Analysis
            </h3>
            <p className="text-gray-400">
              Compare your measurements against thousands of elite athletes to find your perfect match
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05, y: -10 }}
            className="card card-hover text-center p-8"
          >
            <Zap className="mx-auto mb-4 text-neon-green" size={48} />
            <h3 className="text-xl font-montserrat font-bold mb-3 text-neon-green">
              Instant Results
            </h3>
            <p className="text-gray-400">
              Get your personalized sport recommendations and athlete matches in seconds
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.05, y: -10 }}
            className="card card-hover text-center p-8"
          >
            <Users className="mx-auto mb-4 text-neon-gold" size={48} />
            <h3 className="text-xl font-montserrat font-bold mb-3 text-neon-gold">
              Elite Database
            </h3>
            <p className="text-gray-400">
              Access comprehensive data from world-class athletes across all major sports
            </p>
          </motion.div>
        </motion.div>

        {/* Stats Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1.2 }}
          className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-8 max-w-4xl mx-auto"
        >
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bebas text-neon-blue mb-2">50+</div>
            <div className="text-gray-400">Elite Athletes</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bebas text-neon-green mb-2">8</div>
            <div className="text-gray-400">Sports Analyzed</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bebas text-neon-gold mb-2">6</div>
            <div className="text-gray-400">Body Measurements</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bebas text-neon-blue mb-2">95%</div>
            <div className="text-gray-400">Accuracy Rate</div>
          </div>
        </motion.div>
      </div>

      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{
            rotate: 360,
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-1/4 left-1/4 w-32 h-32 border border-neon-blue/20 rounded-full"
        />
        <motion.div
          animate={{
            rotate: -360,
            scale: [1.2, 1, 1.2],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-3/4 right-1/4 w-24 h-24 border border-neon-green/20 rounded-full"
        />
        <motion.div
          animate={{
            y: [-20, 20, -20],
            x: [-10, 10, -10],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute top-1/2 right-1/3 w-16 h-16 bg-neon-gold/10 rounded-full"
        />
        <motion.div
          animate={{
            y: [20, -20, 20],
            x: [10, -10, 10],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute bottom-1/4 left-1/3 w-20 h-20 bg-neon-blue/10 rounded-full"
        />
      </div>
    </div>
  );
}
