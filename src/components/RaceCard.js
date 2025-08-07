import React from 'react';
import { motion } from 'framer-motion';

const RaceCard = ({ race, onEdit, onDelete }) => {
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  const getCountryFlag = (country) => {
    const flags = {
      'Bahrain': '🇧🇭',
      'Saudi Arabia': '🇸🇦',
      'Australia': '🇦🇺',
      'Japan': '🇯🇵',
      'China': '🇨🇳',
      'USA': '🇺🇸',
      'Monaco': '🇲🇨',
      'Spain': '🇪🇸',
      'Canada': '🇨🇦',
      'Austria': '🇦🇹',
      'United Kingdom': '🇬🇧',
      'Hungary': '🇭🇺',
      'Belgium': '🇧🇪',
      'Netherlands': '🇳🇱',
      'Italy': '🇮🇹',
      'Singapore': '🇸🇬',
      'Azerbaijan': '🇦🇿',
      'Qatar': '🇶🇦',
      'Mexico': '🇲🇽',
      'Brazil': '🇧🇷',
      'UAE': '🇦🇪',
      'France': '🇫🇷',
      'Germany': '🇩🇪',
      'Portugal': '🇵🇹',
      'Turkey': '🇹🇷',
      'Russia': '🇷🇺'
    };
    return flags[country] || '🏁';
  };

  const isUpcoming = () => {
    const today = new Date();
    const raceDate = new Date(race.date);
    return raceDate > today;
  };

  const getDaysUntilRace = () => {
    const today = new Date();
    const raceDate = new Date(race.date);
    const diffTime = raceDate - today;
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  const daysUntil = getDaysUntilRace();
  const upcoming = isUpcoming();

  return (
    <motion.div
      className="race-card group"
      whileHover={{ y: -5 }}
      whileTap={{ scale: 0.98 }}
    >
      {/* Status Badge */}
      <div className="flex justify-between items-start mb-4">
        <motion.div 
          className={`px-3 py-1 rounded-full text-xs font-bold ${
            upcoming 
              ? 'bg-green-500 text-white' 
              : 'bg-gray-500 text-gray-200'
          }`}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 500 }}
        >
          {upcoming ? `${daysUntil} days` : 'Completed'}
        </motion.div>
        
        <div className="flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
          <motion.button
            onClick={() => onEdit(race)}
            className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            ✏️
          </motion.button>
          <motion.button
            onClick={() => onDelete(race.id)}
            className="p-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors duration-200"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
          >
            🗑️
          </motion.button>
        </div>
      </div>

      {/* Race Header */}
      <div className="flex items-center mb-3">
        <motion.div 
          className="text-3xl mr-3"
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
        >
          {getCountryFlag(race.country)}
        </motion.div>
        <div>
          <h3 className="text-xl font-bold text-white mb-1">{race.name}</h3>
          <p className="text-gray-300 text-sm">{race.country}</p>
        </div>
      </div>

      {/* Location */}
      <div className="mb-4">
        <div className="flex items-center text-gray-300 mb-2">
          <span className="mr-2">📍</span>
          <span className="font-medium">{race.location}</span>
        </div>
      </div>

      {/* Date */}
      <div className="mb-4">
        <div className="flex items-center text-gray-300 mb-2">
          <span className="mr-2">📅</span>
          <span>{formatDate(race.date)}</span>
        </div>
      </div>

      {/* Animated Border */}
      <motion.div
        className="absolute inset-0 rounded-lg bg-gradient-to-r from-f1-red to-red-700 opacity-0 group-hover:opacity-10 transition-opacity duration-300"
        initial={{ scale: 0.9 }}
        whileHover={{ scale: 1 }}
        transition={{ duration: 0.3 }}
      />

      {/* Live Indicator for Upcoming Races */}
      {upcoming && daysUntil <= 7 && (
        <motion.div
          className="absolute top-2 right-2 w-3 h-3 bg-f1-red rounded-full"
          animate={{ 
            scale: [1, 1.3, 1],
            opacity: [0.7, 1, 0.7]
          }}
          transition={{ 
            duration: 1.5, 
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      )}
    </motion.div>
  );
};

export default RaceCard;
