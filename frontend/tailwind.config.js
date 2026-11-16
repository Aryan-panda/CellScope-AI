/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: '#0B1120',
        surface: '#070F1C',
        panel: '#111B2E',
        card: '#182438',
        border: '#29364A',
        teal: {
          DEFAULT: '#2DD4BF',
          light: '#5EEAD4',
          dark: '#14B8A6',
          glow: 'rgba(45,212,191,0.15)',
        },
        blue: {
          DEFAULT: '#60A5FA',
          light: '#93C5FD',
          dark: '#3B82F6',
          glow: 'rgba(96,165,250,0.15)',
        },
        rose: {
          DEFAULT: '#FB7185',
          light: '#FDA4AF',
          dark: '#F43F5E',
        },
        'slate-custom': {
          100: '#F1F5F9',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
