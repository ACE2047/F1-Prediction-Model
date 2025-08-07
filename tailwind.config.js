/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        f1: {
          red: '#E10600',
          dark: '#15151E',
          silver: '#C0C0C0',
          gold: '#FFD700'
        }
      },
      animation: {
        'slide-in': 'slideIn 0.5s ease-out',
        'fade-in': 'fadeIn 0.3s ease-in',
        'bounce-in': 'bounceIn 0.6s ease-out',
        'pulse-slow': 'pulse 3s infinite'
      },
      keyframes: {
        slideIn: {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' }
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' }
        },
        bounceIn: {
          '0%': { transform: 'scale(0.3)', opacity: '0' },
          '50%': { transform: 'scale(1.05)' },
          '70%': { transform: 'scale(0.9)' },
          '100%': { transform: 'scale(1)', opacity: '1' }
        }
      }
    },
  },
  plugins: [],
}
