import type { Metadata } from 'next'
import { Inter, Bebas_Neue, Montserrat, Poppins, Oswald } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' })
const bebas = Bebas_Neue({ 
  weight: '400', 
  subsets: ['latin'], 
  variable: '--font-bebas' 
})
const montserrat = Montserrat({ 
  subsets: ['latin'], 
  variable: '--font-montserrat' 
})
const poppins = Poppins({ 
  weight: ['300', '400', '500', '600', '700'], 
  subsets: ['latin'], 
  variable: '--font-poppins' 
})
const oswald = Oswald({ 
  weight: ['300', '400', '500', '600', '700'], 
  subsets: ['latin'], 
  variable: '--font-oswald' 
})

export const metadata: Metadata = {
  title: 'Hall of Frame - Find Your Sport',
  description: 'Discover which sport you are best suited for based on your body measurements',
  keywords: 'sports, athlete, body measurements, fitness, recommendations',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${bebas.variable} ${montserrat.variable} ${poppins.variable} ${oswald.variable}`}>
      <body className="min-h-screen text-text-primary relative overflow-x-hidden">
        {/* Bold floating background elements */}
        <div className="floating-bg">
          <div className="floating-shape bg-gradient-to-br from-neon-blue/20 to-neon-pink/10"></div>
          <div className="floating-shape bg-gradient-to-br from-neon-pink/20 to-neon-green/10"></div>
          <div className="floating-shape bg-gradient-to-br from-neon-green/20 to-neon-orange/10"></div>
          <div className="floating-shape bg-gradient-to-br from-neon-orange/20 to-neon-blue/10"></div>
        </div>
        {children}
      </body>
    </html>
  )
}
