'use client';

import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Activity, Zap, Target, Trophy, BarChart3, Users } from 'lucide-react';

const loadingMessages = [
  { text: 'Analyzing your measurements...', icon: BarChart3, delay: 0 },
  { text: 'Comparing against elite athletes...', icon: Users, delay: 1000 },
  { text: 'Calculating similarity scores...', icon: Target, delay: 2000 },
  { text: 'Finding your perfect sport match...', icon: Trophy, delay: 3000 },
  { text: 'Generating personalized results...', icon: Zap, delay: 4000 },
  { text: 'Almost ready...', icon: Activity, delay: 5000 }
];

export default function LoadingPage() {
  const router = useRouter();
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // Auto-redirect to results after 6 seconds
    const redirectTimer = setTimeout(() => {
      router.push('/results');
    }, 6000);

    // Progress animation
    const progressTimer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressTimer);
          return 100;
        }
        return prev + 1.5;
      });
    }, 90);

    // Message rotation
    const messageTimer = setInterval(() => {
      setCurrentMessageIndex(prev => (prev + 1) % loadingMessages.length);
    }, 1000);

    return () => {
      clearTimeout(redirectTimer);
      clearInterval(progressTimer);
      clearInterval(messageTimer);
    };
  }, [router]);

  const currentMessage = loadingMessages[currentMessageIndex];
  const IconComponent = currentMessage.icon;

  return (
    <div className="min-h-screen bg-dark-bg flex items-center justify-center px-4">
      <div className="max-w-2xl mx-auto text-center">
        {/* Main Loading Animation */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
          className="mb-12"
        >
          {/* Central Loading Circle */}
          <div className="relative w-32 h-32 mx-auto mb-8">
            <motion.div
              className="absolute inset-0 border-4 border-neon-blue/20 rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute inset-2 border-4 border-neon-green/20 rounded-full"
              animate={{ rotate: -360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            />
            <motion.div
              className="absolute inset-4 border-4 border-neon-gold/20 rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Center Icon */}
            <motion.div
              className="absolute inset-0 flex items-center justify-center"
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <Trophy className="text-neon-blue" size={48} />
            </motion.div>
          </div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-4xl md:text-5xl font-bebas gradient-text mb-4"
          >
            ANALYZING YOUR FRAME
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-xl text-gray-400 mb-8"
          >
            Finding your perfect athletic match...
          </motion.p>
        </motion.div>

        {/* Progress Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mb-8"
        >
          <div className="w-full bg-gray-800 rounded-full h-3 mb-4 overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-neon-blue via-neon-green to-neon-gold rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.1 }}
            />
          </div>
          <div className="flex justify-between text-sm text-gray-400">
            <span>Processing...</span>
            <span>{Math.round(progress)}%</span>
          </div>
        </motion.div>

        {/* Current Message */}
        <motion.div
          key={currentMessageIndex}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.5 }}
          className="flex items-center justify-center gap-4 mb-8"
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <IconComponent className="text-neon-blue" size={24} />
          </motion.div>
          <span className="text-lg font-montserrat text-white">
            {currentMessage.text}
          </span>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4"
        >
          <motion.div
            className="card p-4 text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bebas text-neon-blue mb-1">50+</div>
            <div className="text-xs text-gray-400">Athletes</div>
          </motion.div>
          <motion.div
            className="card p-4 text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bebas text-neon-green mb-1">8</div>
            <div className="text-xs text-gray-400">Sports</div>
          </motion.div>
          <motion.div
            className="card p-4 text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bebas text-neon-gold mb-1">6</div>
            <div className="text-xs text-gray-400">Measurements</div>
          </motion.div>
          <motion.div
            className="card p-4 text-center"
            whileHover={{ scale: 1.05 }}
          >
            <div className="text-2xl font-bebas text-neon-blue mb-1">95%</div>
            <div className="text-xs text-gray-400">Accuracy</div>
          </motion.div>
        </motion.div>

        {/* Animated Background Elements */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10">
          <motion.div
            animate={{
              x: [0, 100, 0],
              y: [0, -50, 0],
              rotate: [0, 180, 360],
            }}
            transition={{
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute top-1/4 left-1/4 w-20 h-20 border border-neon-blue/20 rounded-full"
          />
          <motion.div
            animate={{
              x: [0, -80, 0],
              y: [0, 60, 0],
              rotate: [360, 180, 0],
            }}
            transition={{
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute top-3/4 right-1/4 w-16 h-16 border border-neon-green/20 rounded-full"
          />
          <motion.div
            animate={{
              x: [0, 60, 0],
              y: [0, -40, 0],
              scale: [1, 1.2, 1],
            }}
            transition={{
              duration: 6,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="absolute top-1/2 left-1/3 w-12 h-12 bg-neon-gold/10 rounded-full"
          />
        </div>
      </div>
    </div>
  );
}
