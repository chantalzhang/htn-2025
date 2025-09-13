import type { Metadata } from 'next'
import { Inter, Bebas_Neue, Montserrat, Poppins } from 'next/font/google'
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
    <html lang="en" className={`${inter.variable} ${bebas.variable} ${montserrat.variable} ${poppins.variable}`}>
      <body className="min-h-screen bg-dark-bg text-white">
        <div className="floating-bg">
          <div className="floating-shape"></div>
          <div className="floating-shape"></div>
          <div className="floating-shape"></div>
          <div className="floating-shape"></div>
        </div>
        {children}
      </body>
    </html>
  )
}
