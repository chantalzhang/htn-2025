/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Bold OpenAI-inspired color palette
        'primary-blue': '#1e40af', // Deep royal blue
        'accent-blue': '#3b82f6', // Bright blue
        'neon-blue': '#00d4ff', // Electric cyan
        'primary-pink': '#ec4899', // Vibrant pink
        'accent-pink': '#f472b6', // Light pink
        'neon-pink': '#ff0080', // Hot pink
        'primary-green': '#10b981', // Emerald green
        'accent-green': '#34d399', // Light green
        'neon-green': '#00ff88', // Electric green
        'primary-orange': '#f97316', // Orange
        'accent-orange': '#fb923c', // Light orange
        'neon-orange': '#ff6b35', // Bright orange
        'dark-bg': '#0c0c0c', // Deep black
        'dark-card': '#1a1a1a', // Card background
        'text-primary': '#ffffff', // Pure white
        'text-secondary': '#e5e7eb', // Light gray
        'text-muted': '#9ca3af', // Muted gray
      },
      fontFamily: {
        'bebas': ['Bebas Neue', 'cursive'],
        'montserrat': ['Montserrat', 'sans-serif'],
        'inter': ['Inter', 'sans-serif'],
        'poppins': ['Poppins', 'sans-serif'],
        'oswald': ['Oswald', 'sans-serif'],
      },
      animation: {
        'float': 'float 6s ease-in-out infinite',
        'pulse-neon': 'pulse-neon 2s ease-in-out infinite',
        'slide-up': 'slide-up 0.5s ease-out',
        'fade-in': 'fade-in 0.5s ease-out',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'pulse-neon': {
          '0%, 100%': { 
            boxShadow: '0 0 5px currentColor, 0 0 10px currentColor, 0 0 15px currentColor',
            opacity: '1'
          },
          '50%': { 
            boxShadow: '0 0 10px currentColor, 0 0 20px currentColor, 0 0 30px currentColor',
            opacity: '0.8'
          },
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
