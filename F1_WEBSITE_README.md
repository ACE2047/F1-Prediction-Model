# Formula 1 Website Clone

A comprehensive Formula 1 website clone built with React, featuring modern design, smooth animations, and authentic F1 branding. This project replicates the official Formula 1 website's layout, functionality, and visual appeal.

## 🏎️ Features

### Core Sections
- **Home Page**: Hero carousel, race schedule preview, latest news, drivers showcase, and teams overview
- **Schedule**: Complete 2025 race calendar with race details, status indicators, and statistics
- **Results/Standings**: Driver and constructor championship standings with interactive visualizations
- **News**: Latest F1 news with category filtering, featured articles, and detailed article modals
- **Drivers**: Comprehensive driver profiles with stats, biographies, and career highlights
- **Teams**: Detailed team information including technical specifications and driver lineups

### Design & User Experience
- **Authentic F1 Branding**: Official color schemes, typography, and visual elements
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Smooth Animations**: Framer Motion powered transitions and micro-interactions
- **Interactive Elements**: Hover effects, modal dialogs, and dynamic content
- **Modern UI**: Clean, professional interface matching the official F1 website

### Technical Features
- **React 18**: Latest React features with functional components and hooks
- **Tailwind CSS**: Utility-first CSS framework for rapid styling
- **Framer Motion**: Advanced animations and page transitions
- **Component Architecture**: Modular, reusable components
- **State Management**: React hooks for local state management
- **Performance Optimized**: Lazy loading and optimized rendering

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd f1-prediction-model
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000` to view the website

### Build for Production
```bash
npm run build
```

## 📁 Project Structure

```
src/
├── components/
│   ├── F1Website.js          # Main website component
│   ├── ScheduleSection.js    # Race calendar section
│   ├── StandingsSection.js   # Championship standings
│   ├── NewsSection.js        # News and articles
│   ├── DriversSection.js     # Driver profiles
│   ├── TeamsSection.js       # Team information
│   └── ...
├── data/
│   └── raceData.js          # Race and championship data
├── index.css                # Global styles and Tailwind config
├── index.js                 # Application entry point
└── App.js                   # Root component
```

## 🎨 Design System

### Color Palette
- **Primary Red**: `#dc2626` (F1 Red)
- **Secondary Colors**: Team-specific gradients
- **Background**: Dark theme with gray variations
- **Accent Colors**: Yellow for championships, orange for podiums

### Typography
- **Primary Font**: Inter (Google Fonts)
- **Headings**: Bold weights (600-900)
- **Body Text**: Regular and medium weights (400-500)

### Components
- **Cards**: Rounded corners, hover effects, gradient backgrounds
- **Buttons**: Primary (red), secondary (gray), outline variants
- **Modals**: Full-screen overlays with detailed content
- **Navigation**: Sticky header with smooth transitions

## 📊 Data Structure

### Race Calendar
```javascript
{
  round: 1,
  name: "Bahrain Grand Prix",
  location: "Bahrain International Circuit",
  country: "Bahrain",
  flag: "🇧🇭",
  date: "March 2, 2025",
  status: "completed|next|upcoming",
  winner: "Driver Name",
  image: "race-image-url"
}
```

### Driver Information
```javascript
{
  name: "Driver Name",
  team: "Team Name",
  nationality: "Country",
  points: 476,
  wins: 4,
  podiums: 12,
  position: 1,
  image: "driver-image-url",
  bio: "Driver biography..."
}
```

### Team Details
```javascript
{
  name: "Team Name",
  base: "Location",
  teamPrincipal: "Principal Name",
  points: 910,
  championships: 8,
  drivers: [...],
  technicalSpecs: {...}
}
```

## 🔧 Customization

### Adding New Sections
1. Create a new component in `src/components/`
2. Import and add to the main `F1Website.js` component
3. Update navigation items array
4. Add routing logic in the main component

### Modifying Styles
- Edit `src/index.css` for global styles
- Use Tailwind classes for component-specific styling
- Customize color palette in Tailwind config

### Updating Data
- Modify data files in `src/data/`
- Update component props and state
- Ensure data structure consistency

## 🌟 Key Features Breakdown

### Home Page
- **Hero Carousel**: Auto-advancing slides with featured content
- **Race Schedule**: Next/previous race indicators with quick navigation
- **Latest News**: Grid layout with category badges and timestamps
- **Drivers Showcase**: Team-colored cards with driver stats
- **Teams Overview**: Logo and car displays with hover effects

### Schedule Section
- **Complete Calendar**: All 24 races with detailed information
- **Status Indicators**: Visual distinction between completed, next, and upcoming races
- **Statistics Panel**: Season overview with key metrics
- **Interactive Cards**: Hover effects and detailed race information

### Standings Section
- **Dual View**: Toggle between driver and constructor standings
- **Position Indicators**: Gold, silver, bronze for podium positions
- **Progress Bars**: Visual representation of championship battle
- **Detailed Stats**: Points, wins, podiums for each competitor

### News Section
- **Category Filtering**: Filter by news type (Unlocked, Race, Driver, etc.)
- **Featured Articles**: Highlighted stories with larger display
- **Article Modals**: Full-screen reading experience
- **Newsletter Signup**: Email subscription component

### Drivers Section
- **Team Filtering**: Filter drivers by team affiliation
- **Detailed Profiles**: Comprehensive driver information
- **Career Highlights**: Key achievements and milestones
- **Interactive Cards**: Hover effects and modal dialogs

### Teams Section
- **Constructor Focus**: Detailed team information and history
- **Technical Specs**: Car and engine specifications
- **Driver Lineups**: Current season driver pairings
- **Championship Battle**: Visual progress comparison

## 🎯 Performance Optimizations

- **Lazy Loading**: Images and components loaded on demand
- **Optimized Animations**: Hardware-accelerated CSS transforms
- **Efficient Rendering**: React.memo and useMemo where appropriate
- **Compressed Assets**: Optimized images and fonts
- **Code Splitting**: Dynamic imports for large components

## 📱 Responsive Design

- **Mobile First**: Designed for mobile devices first
- **Breakpoints**: Tailored layouts for different screen sizes
- **Touch Friendly**: Optimized for touch interactions
- **Performance**: Lightweight on mobile networks

## 🔮 Future Enhancements

- **Live Timing**: Real-time race data integration
- **User Accounts**: Personalized experience and preferences
- **Fantasy Integration**: F1 Fantasy league features
- **Social Features**: Comments, sharing, and community interaction
- **PWA Support**: Progressive Web App capabilities
- **Multi-language**: Internationalization support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational and demonstration purposes. Formula 1 trademarks and content are owned by Formula One World Championship Limited.

## 🙏 Acknowledgments

- **Formula 1**: Official website design inspiration
- **React Team**: Amazing framework and documentation
- **Tailwind CSS**: Excellent utility-first CSS framework
- **Framer Motion**: Smooth animation library
- **Community**: Open source contributors and F1 fans

---

**Note**: This is a clone/replica project created for educational purposes. All Formula 1 trademarks, logos, and content are property of Formula One World Championship Limited.