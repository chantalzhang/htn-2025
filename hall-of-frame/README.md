# 🏆 Hall of Frame

**Does your frame belong in the Hall of Fame?**

A responsive web application that helps users discover which sport they are best suited for based on their body measurements, by comparing them against an elite athlete dataset.

## ✨ Features

- **Interactive Body Silhouette**: Click on different body parts to enter measurements with hover animations
- **Elite Athlete Database**: Compare against 50+ world-class athletes across 8 different sports
- **Smart Recommendations**: Get personalized sport recommendations with compatibility scores
- **Athlete Matching**: Find your closest athletic match with detailed similarity analysis
- **Responsive Design**: Beautiful dark theme with neon accents, optimized for all devices
- **Smooth Animations**: Framer Motion powered transitions and micro-interactions
- **Share Results**: Generate shareable results to show off your athletic potential

## 🚀 Tech Stack

- **Framework**: Next.js 14 with App Router
- **Language**: TypeScript
- **Styling**: TailwindCSS with custom neon theme
- **Animations**: Framer Motion
- **Icons**: Lucide React
- **Fonts**: Bebas Neue, Montserrat, Inter, Poppins

## 🎯 Sports Analyzed

- 🏀 Basketball
- 🏈 Football
- ⚽ Soccer
- 🎾 Tennis
- 🏃 Track & Field
- 🏊 Swimming
- 🥊 MMA
- 🤸 Gymnastics

## 📊 Measurements Tracked

- **Height** (cm)
- **Weight** (kg)
- **Wingspan** (cm)
- **Shoulder Width** (cm) - Optional
- **Waist** (cm) - Optional
- **Hip** (cm) - Optional

## 🏃‍♂️ Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hall-of-frame
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Build for Production

```bash
npm run build
npm start
```

## 🎨 Design System

### Color Palette
- **Dark Background**: `#0a0a0a`
- **Card Background**: `#1a1a1a`
- **Neon Blue**: `#00f5ff`
- **Neon Green**: `#39ff14`
- **Neon Gold**: `#ffd700`

### Typography
- **Headers**: Bebas Neue (Bold, Impact)
- **Subheaders**: Montserrat (Bold, Clean)
- **Body**: Inter (Readable, Modern)
- **Accent**: Poppins (Friendly, Approachable)

### Animations
- **Page Transitions**: Smooth fade and slide effects
- **Hover Effects**: Scale and glow animations
- **Loading States**: Rotating elements with progress bars
- **Micro-interactions**: Button presses and form interactions

## 📱 Pages

### 1. Landing Page (`/`)
- Hero section with animated background
- Feature highlights
- Call-to-action to start the journey

### 2. Input Page (`/input`)
- Interactive body silhouette
- Manual measurement inputs
- Progress tracking
- Measurement tips

### 3. Loading Page (`/loading`)
- Sports-style progress animation
- Rotating status messages
- Auto-redirect to results

### 4. Results Page (`/results`)
- Hero reveal with top athlete match
- Top 3 sport recommendations
- Top 3 athlete matches
- Detailed analysis and insights
- Share functionality

## 🧮 Algorithm

The app uses a sophisticated similarity calculation algorithm:

1. **Weighted Measurements**: Different body measurements have different importance weights
2. **Sport-Specific Scoring**: Each sport has unique criteria for optimal body types
3. **Similarity Calculation**: Uses percentage-based similarity scoring
4. **Trait Matching**: Identifies specific matching physical traits
5. **Recommendation Engine**: Generates personalized sport recommendations

## 🎯 Athlete Database

The app includes data for 50+ elite athletes including:

- **Basketball**: LeBron James, Stephen Curry, Kobe Bryant
- **Football**: Tom Brady
- **Soccer**: Lionel Messi, Cristiano Ronaldo, Alex Morgan
- **Tennis**: Serena Williams
- **Track & Field**: Usain Bolt
- **Swimming**: Michael Phelps
- **MMA**: Conor McGregor
- **Gymnastics**: Simone Biles

Each athlete profile includes:
- Physical measurements
- Sport and position
- Key achievements
- Career highlights

## 🔧 Customization

### Adding New Athletes

Edit `src/data/athletes.ts` to add new athletes:

```typescript
{
  id: 'unique-id',
  name: 'Athlete Name',
  sport: 'Sport',
  position: 'Position',
  height: 180, // cm
  weight: 75,  // kg
  wingspan: 185, // cm
  // ... other measurements
  achievements: ['Achievement 1', 'Achievement 2']
}
```

### Adding New Sports

Edit `src/data/athletes.ts` to add new sports:

```typescript
{
  name: 'Sport Name',
  icon: '🏆',
  description: 'Sport description',
  keyTraits: ['trait1', 'trait2', 'trait3']
}
```

### Modifying Scoring Algorithm

Edit `src/utils/similarity.ts` to adjust:
- Measurement weights
- Sport-specific scoring criteria
- Similarity calculation methods

## 📱 Responsive Design

The app is fully responsive with:
- **Mobile-first approach**
- **Breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Touch-friendly interactions**
- **Optimized layouts for all screen sizes**

## 🎨 Animation Details

### Framer Motion Usage
- **Page transitions**: Smooth entrance/exit animations
- **Staggered animations**: Sequential element reveals
- **Hover effects**: Interactive feedback
- **Loading states**: Engaging progress indicators

### Custom CSS Animations
- **Floating background elements**
- **Neon glow effects**
- **Gradient text animations**
- **Progress bar animations**

## 🚀 Performance

- **Next.js optimization**: Automatic code splitting and optimization
- **Image optimization**: Next.js Image component
- **Font optimization**: Google Fonts with display swap
- **Bundle analysis**: Optimized bundle size

## 🔒 Privacy

- **No data collection**: All measurements are processed locally
- **No external APIs**: Complete client-side processing
- **Local storage**: Measurements stored temporarily in browser
- **No tracking**: Privacy-first approach

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🎉 Acknowledgments

- **Athletes**: Thanks to all the incredible athletes who inspire us
- **Design Inspiration**: Nike campaigns and ESPN Draft Combine aesthetics
- **Icons**: Lucide React icon library
- **Fonts**: Google Fonts

---

**Built with ❤️ for sports enthusiasts and aspiring athletes**

*Find your athletic potential and discover which sport you're built for!*
