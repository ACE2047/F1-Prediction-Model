import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import RaceCard from './components/RaceCard';
import AddRaceModal from './components/AddRaceModal';
import EditRaceModal from './components/EditRaceModal';

const initialRaces = [
  { id: 1, name: 'Bahrain Grand Prix', location: 'Bahrain International Circuit', country: 'Bahrain', date: '2025-03-16' },
  { id: 2, name: 'Saudi Arabian Grand Prix', location: 'Jeddah Corniche Circuit', country: 'Saudi Arabia', date: '2025-03-23' },
  { id: 3, name: 'Australian Grand Prix', location: 'Albert Park Circuit', country: 'Australia', date: '2025-04-06' },
  { id: 4, name: 'Chinese Grand Prix', location: 'Shanghai International Circuit', country: 'China', date: '2025-04-20' },
  { id: 5, name: 'Japanese Grand Prix', location: 'Suzuka International Racing Course', country: 'Japan', date: '2025-05-04' },
  { id: 6, name: 'Miami Grand Prix', location: 'Miami International Autodrome', country: 'USA', date: '2025-05-18' },
  { id: 7, name: 'Spanish Grand Prix', location: 'Circuit de Barcelona-Catalunya', country: 'Spain', date: '2025-06-01' },
  { id: 8, name: 'Monaco Grand Prix', location: 'Circuit de Monaco', country: 'Monaco', date: '2025-06-15' },
  { id: 9, name: 'Canadian Grand Prix', location: 'Circuit Gilles Villeneuve', country: 'Canada', date: '2025-06-29' },
  { id: 10, name: 'Austrian Grand Prix', location: 'Red Bull Ring', country: 'Austria', date: '2025-07-13' },
  { id: 11, name: 'British Grand Prix', location: 'Silverstone Circuit', country: 'United Kingdom', date: '2025-07-27' },
  { id: 12, name: 'Belgian Grand Prix', location: 'Circuit de Spa-Francorchamps', country: 'Belgium', date: '2025-08-10' },
  { id: 13, name: 'Dutch Grand Prix', location: 'Circuit Zandvoort', country: 'Netherlands', date: '2025-08-24' },
  { id: 14, name: 'Italian Grand Prix', location: 'Autodromo Nazionale di Monza', country: 'Italy', date: '2025-09-07' },
  { id: 15, name: 'Singapore Grand Prix', location: 'Marina Bay Street Circuit', country: 'Singapore', date: '2025-09-21' },
  { id: 16, name: 'United States Grand Prix', location: 'Circuit of the Americas', country: 'USA', date: '2025-10-05' },
  { id: 17, name: 'Mexican Grand Prix', location: 'Autódromo Hermanos Rodríguez', country: 'Mexico', date: '2025-10-19' },
  { id: 18, name: 'Brazilian Grand Prix', location: 'Interlagos Circuit', country: 'Brazil', date: '2025-11-02' },
  { id: 19, name: 'Las Vegas Grand Prix', location: 'Las Vegas Strip Circuit', country: 'USA', date: '2025-11-16' },
  { id: 20, name: 'Qatar Grand Prix', location: 'Lusail International Circuit', country: 'Qatar', date: '2025-11-30' },
  { id: 21, name: 'Abu Dhabi Grand Prix', location: 'Yas Marina Circuit', country: 'UAE', date: '2025-12-14' }
];

function App() {
  const [races, setRaces] = useState(initialRaces);
  const [showAddModal, setShowAddModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingRace, setEditingRace] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update time every second for auto-updating effect
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const addRace = (newRace) => {
    const race = {
      id: Math.max(...races.map(r => r.id)) + 1,
      ...newRace
    };
    setRaces([...races, race]);
  };

  const updateRace = (updatedRace) => {
    setRaces(races.map(race => 
      race.id === updatedRace.id ? updatedRace : race
    ));
  };

  const deleteRace = (raceId) => {
    setRaces(races.filter(race => race.id !== raceId));
  };

  const openEditModal = (race) => {
    setEditingRace(race);
    setShowEditModal(true);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-f1-dark">
      {/* Header */}
      <motion.header 
        className="bg-gradient-to-r from-f1-red to-red-700 shadow-2xl"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        <div className="container mx-auto px-6 py-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">
                🏎️ F1 Race Location Manager
              </h1>
              <div className="flex items-center space-x-3 mb-2">
                <p className="text-red-100">
                  2025 Formula 1 World Championship
                </p>
                <motion.div
                  className="bg-f1-gold text-black px-3 py-1 rounded-full text-sm font-bold"
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  SEASON 2025
                </motion.div>
              </div>
              <p className="text-red-200 text-sm">
                Manage race locations with real-time updates • {races.length} races scheduled
              </p>
            </div>
            <div className="text-right">
              <motion.div 
                className="text-white text-lg font-mono"
                key={currentTime.toLocaleTimeString()}
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.2 }}
              >
                {currentTime.toLocaleString()}
              </motion.div>
              <div className="text-red-100 text-sm">
                Live Updates Active
              </div>
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Controls */}
        <motion.div 
          className="mb-8 flex justify-between items-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.6 }}
        >
          <div className="flex items-center space-x-4">
            <h2 className="text-2xl font-bold text-white">
              Race Calendar ({races.length} races)
            </h2>
            <motion.div 
              className="w-3 h-3 bg-green-500 rounded-full animate-pulse-slow"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
          <button
            onClick={() => setShowAddModal(true)}
            className="btn-primary flex items-center space-x-2"
          >
            <span>➕</span>
            <span>Add New Race</span>
          </button>
        </motion.div>

        {/* Race Grid */}
        <motion.div 
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 0.6 }}
        >
          <AnimatePresence mode="popLayout">
            {races.map((race, index) => (
              <motion.div
                key={race.id}
                layout
                initial={{ opacity: 0, scale: 0.8, y: 50 }}
                animate={{ 
                  opacity: 1, 
                  scale: 1, 
                  y: 0,
                  transition: { delay: index * 0.1, duration: 0.5 }
                }}
                exit={{ 
                  opacity: 0, 
                  scale: 0.8, 
                  y: -50,
                  transition: { duration: 0.3 }
                }}
                whileHover={{ 
                  scale: 1.02,
                  transition: { duration: 0.2 }
                }}
              >
                <RaceCard
                  race={race}
                  onEdit={openEditModal}
                  onDelete={deleteRace}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>

        {/* Empty State */}
        {races.length === 0 && (
          <motion.div 
            className="text-center py-16"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
          >
            <div className="text-6xl mb-4">🏁</div>
            <h3 className="text-2xl font-bold text-gray-300 mb-2">No races scheduled</h3>
            <p className="text-gray-500 mb-6">Add your first F1 race to get started</p>
            <button
              onClick={() => setShowAddModal(true)}
              className="btn-primary"
            >
              Add Your First Race
            </button>
          </motion.div>
        )}
      </main>

      {/* Footer */}
      <motion.footer 
        className="bg-gray-800 border-t border-gray-700 mt-16"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
      >
        <div className="container mx-auto px-6 py-6 text-center text-gray-400">
          <p>F1 Race Location Manager | Built with React & Tailwind CSS</p>
          <div className="flex justify-center items-center mt-2 space-x-2">
            <span>Updates every second</span>
            <motion.div 
              className="w-2 h-2 bg-f1-red rounded-full"
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
          </div>
        </div>
      </motion.footer>

      {/* Modals */}
      <AddRaceModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAdd={addRace}
      />

      <EditRaceModal
        isOpen={showEditModal}
        onClose={() => setShowEditModal(false)}
        race={editingRace}
        onUpdate={updateRace}
      />
    </div>
  );
}

export default App;
