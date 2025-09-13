'use client';

import { motion } from 'framer-motion';
import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import { BodyMeasurements } from '@/types';

interface ValidationIssue {
  field: keyof BodyMeasurements;
  value: number;
  min: number;
  max: number;
  label: string;
}

export default function ValidationPage() {
  const router = useRouter();
  const [measurements, setMeasurements] = useState<BodyMeasurements | null>(null);
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [loading, setLoading] = useState(true);

  // Define expected ranges
  const validationRanges = {
    height: { min: 100, max: 250, label: 'Height' },
    weight: { min: 30, max: 200, label: 'Weight' },
    wingspan: { min: 100, max: 250, label: 'Wingspan' },
    shoulderWidth: { min: 30, max: 80, label: 'Shoulder Width' },
    waist: { min: 50, max: 150, label: 'Waist' },
    hip: { min: 60, max: 160, label: 'Hip' }
  };

  useEffect(() => {
    // Load measurements from localStorage
    const storedMeasurements = localStorage.getItem('userMeasurements');
    if (storedMeasurements) {
      const parsedMeasurements = JSON.parse(storedMeasurements);
      setMeasurements(parsedMeasurements);
      
      // Check for validation issues
      const issues: ValidationIssue[] = [];
      
      Object.entries(parsedMeasurements).forEach(([key, value]) => {
        const field = key as keyof BodyMeasurements;
        const range = validationRanges[field];
        
        if (value > 0 && (value < range.min || value > range.max)) {
          issues.push({
            field,
            value: Number(value),
            min: range.min,
            max: range.max,
            label: range.label
          });
        }
      });
      
      setValidationIssues(issues);
    }
    
    setLoading(false);
  }, []);

  const handleProceed = () => {
    router.push('/loading');
  };

  const handleGoBack = () => {
    router.push('/input');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-neon-blue mx-auto mb-4"></div>
          <p className="text-text-secondary font-oswald">Validating measurements...</p>
        </div>
      </div>
    );
  }

  if (!measurements || validationIssues.length === 0) {
    // No validation issues, redirect to loading
    router.push('/loading');
    return null;
  }

  return (
    <div className="min-h-screen py-8 px-4 relative overflow-hidden">
      {/* Bold Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-radial from-neon-pink/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-gradient-radial from-neon-blue/15 to-transparent rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-radial from-neon-orange/10 to-transparent rounded-full blur-2xl"></div>
      </div>

      <div className="max-w-4xl mx-auto relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <div className="flex items-center justify-center gap-4 mb-6">
            <button
              onClick={handleGoBack}
              className="btn-secondary p-3 hover:bg-gray-700 transition-all duration-300"
            >
              <ArrowLeft size={20} />
            </button>
            <div className="flex-1"></div>
          </div>

          <motion.div
            initial={{ scale: 0.8 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-neon-orange to-neon-pink rounded-full mb-6"
          >
            <AlertTriangle className="text-white" size={40} />
          </motion.div>

          <h1 className="text-6xl md:text-8xl font-oswald font-black text-transparent bg-clip-text bg-gradient-to-r from-neon-orange to-neon-pink mb-4">
            Uh Oh!
          </h1>
          <p className="text-2xl font-oswald text-text-secondary">
            Are you sure about these measurements?
          </p>
        </motion.div>

        {/* Validation Issues */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="card-orange p-8 mb-8"
        >
          <h2 className="text-3xl font-oswald font-black text-neon-orange mb-6 text-center">
            Measurements Outside Expected Range
          </h2>
          
          <div className="space-y-4">
            {validationIssues.map((issue, index) => (
              <motion.div
                key={issue.field}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                className="bg-dark-card/50 rounded-xl p-4 border border-neon-orange/30"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <XCircle className="text-red-500" size={24} />
                    <div>
                      <h3 className="text-xl font-oswald font-bold text-white">
                        {issue.label}
                      </h3>
                      <p className="text-text-secondary font-oswald">
                        Your value: <span className="text-neon-orange font-bold">{issue.value}{issue.field === 'weight' ? 'kg' : 'cm'}</span>
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-text-secondary font-oswald text-sm">
                      Expected range:
                    </p>
                    <p className="text-neon-green font-oswald font-bold">
                      {issue.min}{issue.field === 'weight' ? 'kg' : 'cm'} - {issue.max}{issue.field === 'weight' ? 'kg' : 'cm'}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <motion.button
            onClick={handleGoBack}
            className="btn-secondary px-8 py-4 text-lg font-oswald font-black flex items-center gap-3 hover:bg-gray-700 transition-all duration-300"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <ArrowLeft size={20} />
            Go Back & Edit
          </motion.button>

          <motion.button
            onClick={handleProceed}
            className="btn-accent px-8 py-4 text-lg font-oswald font-black flex items-center gap-3 neon-glow-green hover:neon-glow-pink transition-all duration-300"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <CheckCircle size={20} />
            Proceed Anyway
          </motion.button>
        </motion.div>

        {/* Warning Message */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="mt-8 text-center"
        >
          <p className="text-text-secondary font-oswald text-sm">
            Proceeding with unusual measurements may affect the accuracy of your results
          </p>
        </motion.div>
      </div>
    </div>
  );
}
