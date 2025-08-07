import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const EditRaceModal = ({ isOpen, onClose, race, onUpdate }) => {
  const [formData, setFormData] = useState({
    name: '',
    location: '',
    country: '',
    date: ''
  });

  const [errors, setErrors] = useState({});

  const countries = [
    'Australia', 'Austria', 'Azerbaijan', 'Bahrain', 'Belgium', 'Brazil',
    'Canada', 'China', 'Hungary', 'Italy', 'Japan', 'Mexico', 'Monaco',
    'Netherlands', 'Qatar', 'Saudi Arabia', 'Singapore', 'Spain', 'UAE',
    'United Kingdom', 'USA'
  ];

  // Update form data when race changes
  useEffect(() => {
    if (race) {
      setFormData({
        name: race.name,
        location: race.location,
        country: race.country,
        date: race.date
      });
      setErrors({});
    }
  }, [race]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Race name is required';
    }
    
    if (!formData.location.trim()) {
      newErrors.location = 'Location is required';
    }
    
    if (!formData.country) {
      newErrors.country = 'Country is required';
    }
    
    if (!formData.date) {
      newErrors.date = 'Date is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (validateForm()) {
      onUpdate({
        ...race,
        ...formData
      });
      setErrors({});
      onClose();
    }
  };

  const handleClose = () => {
    setErrors({});
    onClose();
  };

  return (
    <AnimatePresence>
      {isOpen && race && (
        <>
          {/* Backdrop */}
          <motion.div
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
          />
          
          {/* Modal */}
          <motion.div
            className="fixed inset-0 flex items-center justify-center z-50 p-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className="bg-gray-800 rounded-xl shadow-2xl p-8 w-full max-w-md border border-gray-700"
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              transition={{ type: "spring", duration: 0.5 }}
            >
              {/* Header */}
              <div className="flex justify-between items-center mb-6">
                <motion.h2 
                  className="text-2xl font-bold text-white flex items-center"
                  initial={{ x: -20 }}
                  animate={{ x: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  <span className="mr-3">✏️</span>
                  Edit Race
                </motion.h2>
                <motion.button
                  onClick={handleClose}
                  className="text-gray-400 hover:text-white transition-colors p-2"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  ✕
                </motion.button>
              </div>

              {/* Form */}
              <form onSubmit={handleSubmit} className="space-y-4">
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <label className="block text-white text-sm font-medium mb-2">
                    Race Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={`input-field w-full ${errors.name ? 'border-red-500' : ''}`}
                    placeholder="e.g., Monaco Grand Prix"
                  />
                  {errors.name && (
                    <motion.p 
                      className="text-red-400 text-sm mt-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {errors.name}
                    </motion.p>
                  )}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <label className="block text-white text-sm font-medium mb-2">
                    Circuit/Location
                  </label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    className={`input-field w-full ${errors.location ? 'border-red-500' : ''}`}
                    placeholder="e.g., Circuit de Monaco"
                  />
                  {errors.location && (
                    <motion.p 
                      className="text-red-400 text-sm mt-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {errors.location}
                    </motion.p>
                  )}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  <label className="block text-white text-sm font-medium mb-2">
                    Country
                  </label>
                  <select
                    name="country"
                    value={formData.country}
                    onChange={handleChange}
                    className={`input-field w-full ${errors.country ? 'border-red-500' : ''}`}
                  >
                    <option value="">Select a country</option>
                    {countries.map(country => (
                      <option key={country} value={country} className="bg-gray-800">
                        {country}
                      </option>
                    ))}
                  </select>
                  {errors.country && (
                    <motion.p 
                      className="text-red-400 text-sm mt-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {errors.country}
                    </motion.p>
                  )}
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <label className="block text-white text-sm font-medium mb-2">
                    Race Date
                  </label>
                  <input
                    type="date"
                    name="date"
                    value={formData.date}
                    onChange={handleChange}
                    className={`input-field w-full ${errors.date ? 'border-red-500' : ''}`}
                  />
                  {errors.date && (
                    <motion.p 
                      className="text-red-400 text-sm mt-1"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {errors.date}
                    </motion.p>
                  )}
                </motion.div>

                {/* Location Change Indicator */}
                {race && (formData.location !== race.location || formData.country !== race.country) && (
                  <motion.div
                    className="bg-blue-900 border border-blue-600 rounded-lg p-3"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="flex items-center text-blue-300 text-sm">
                      <span className="mr-2">📍</span>
                      <span>Location will be updated from <strong>{race.location}, {race.country}</strong> to <strong>{formData.location}, {formData.country}</strong></span>
                    </div>
                  </motion.div>
                )}

                {/* Buttons */}
                <motion.div 
                  className="flex space-x-4 pt-6"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                >
                  <motion.button
                    type="button"
                    onClick={handleClose}
                    className="btn-secondary flex-1"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Cancel
                  </motion.button>
                  <motion.button
                    type="submit"
                    className="btn-primary flex-1 flex items-center justify-center space-x-2"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <span>💾</span>
                    <span>Update Race</span>
                  </motion.button>
                </motion.div>
              </form>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default EditRaceModal;
