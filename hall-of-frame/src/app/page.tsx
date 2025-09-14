'use client';

import { motion } from 'framer-motion';
import { useRouter } from 'next/navigation';
import { Trophy, Target, Zap, Users } from 'lucide-react';

export default function LandingPage() {
  const router = useRouter();

  const handleGetStarted = () => {
    router.push('/photo');
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Dynamic Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {/* Animated gradient background */}
        <motion.div 
          className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary-blue/10 via-transparent to-neon-pink/10"
          animate={{
            background: [
              'linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 128, 0.1) 100%)',
              'linear-gradient(135deg, rgba(255, 0, 128, 0.1) 0%, rgba(0, 255, 136, 0.1) 100%)',
              'linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%)'
            ]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        
        {/* Pulsing radial gradients */}
        <motion.div 
          className="absolute top-1/4 right-0 w-96 h-96 bg-gradient-radial from-neon-blue/20 to-transparent rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.2, 0.4, 0.2]
          }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />
        
        <motion.div 
          className="absolute bottom-1/4 left-0 w-80 h-80 bg-gradient-radial from-neon-pink/20 to-transparent rounded-full blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.4, 0.2, 0.4]
          }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        
        <motion.div 
          className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-neon-green/15 to-transparent rounded-full blur-2xl"
          animate={{
            scale: [1, 1.3, 1],
            opacity: [0.15, 0.3, 0.15]
          }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {/* Hero Section */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 text-center p-48">
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
            className="text-7xl md:text-9xl font-oswald mb-8 gradient-text leading-none"
            style={{ 
              textShadow: '0 0 40px rgba(0, 212, 255, 0.5), 0 0 80px rgba(255, 0, 128, 0.3)',
              letterSpacing: '0.05em'
            }}
          >
            HALL OF FRAME
          </motion.h1>

          {/* Tagline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-2xl md:text-3xl font-oswald font-bold text-white mb-8 max-w-3xl mx-auto"
            style={{ textShadow: '0 0 20px rgba(255, 255, 255, 0.3)' }}
          >
            Does your frame belong in the Hall of Fame?
          </motion.p>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl md:text-2xl text-text-secondary mb-16 max-w-4xl mx-auto leading-relaxed font-oswald font-medium"
          >
            Discover which sport you're best suited for by comparing your body measurements 
            against elite athletes from around the world. Find your perfect athletic match!
          </motion.p>

          {/* CTA Button */}
          <button
            
            onClick={handleGetStarted}
            className="btn-primary text-3xl px-16 py-6 font-oswald font-black"
            style={{ 
              background: 'linear-gradient(135deg, #1e40af, #3b82f6, #00d4ff)',
              boxShadow: '0 0 30px rgba(0, 212, 255, 0.4), 0 0 60px rgba(0, 212, 255, 0.2)'
            }}
          >
            <Trophy className="inline-block mr-4" size={40} />
            Find Your Sport
          </button>
        </motion.div>

        {/* Features Section */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 1 }}
          className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-10 max-w-7xl mx-auto"
        >
          <motion.div
            whileHover={{ scale: 1.08, y: -15 }}
            className="card-blue text-center p-10"
          >
            <Target className="mx-auto mb-6 text-neon-blue" size={64} />
            <h3 className="text-2xl font-oswald font-black mb-4 gradient-text-blue">
              Precise Analysis
            </h3>
            <p className="text-text-secondary text-lg leading-relaxed">
              Compare your measurements against elite athletes to find your perfect match
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.08, y: -15 }}
            className="card-pink text-center p-10"
          >
            <Zap className="mx-auto mb-6 text-neon-pink" size={64} />
            <h3 className="text-2xl font-oswald font-black mb-4 gradient-text-pink">
              Instant Results
            </h3>
            <p className="text-text-secondary text-lg leading-relaxed">
              Get your personalized sport recommendations and athlete matches in seconds
            </p>
          </motion.div>

          <motion.div
            whileHover={{ scale: 1.08, y: -15 }}
            className="card-green text-center p-10"
          >
            <Users className="mx-auto mb-6 text-neon-green" size={64} />
            <h3 className="text-2xl font-oswald font-black mb-4 text-neon-green">
              Elite Database
            </h3>
            <p className="text-text-secondary text-lg leading-relaxed">
              Access comprehensive data from world-class athletes across all major sports
            </p>
          </motion.div>
        </motion.div>

        {/* Stats Section */}
        
      </div>

      {/* Bold Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{
            rotate: 360,
            scale: [1, 1.3, 1],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-1/4 left-1/4 w-40 h-40 border-2 border-neon-blue/30 rounded-full"
          style={{ boxShadow: '0 0 40px rgba(0, 212, 255, 0.2)' }}
        />
        <motion.div
          animate={{
            rotate: -360,
            scale: [1.3, 1, 1.3],
          }}
          transition={{
            duration: 30,
            repeat: Infinity,
            ease: "linear"
          }}
          className="absolute top-3/4 right-1/4 w-32 h-32 border-2 border-neon-pink/30 rounded-full"
          style={{ boxShadow: '0 0 40px rgba(255, 0, 128, 0.2)' }}
        />
        <motion.div
          animate={{
            y: [-30, 30, -30],
            x: [-15, 15, -15],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute top-1/2 right-1/3 w-20 h-20 bg-gradient-to-br from-neon-green/20 to-neon-green/5 rounded-full"
          style={{ boxShadow: '0 0 30px rgba(0, 255, 136, 0.3)' }}
        />
        <motion.div
          animate={{
            y: [30, -30, 30],
            x: [15, -15, 15],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          className="absolute bottom-1/4 left-1/3 w-24 h-24 bg-gradient-to-br from-neon-orange/20 to-neon-orange/5 rounded-full"
          style={{ boxShadow: '0 0 30px rgba(255, 107, 53, 0.3)' }}
        />
      </div>
    </div>
  );
}
