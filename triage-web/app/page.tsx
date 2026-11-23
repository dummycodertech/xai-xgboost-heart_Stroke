import { FieldBackground } from '@/components/FieldBackground'
import { ThresholdProvider } from '@/components/ThresholdContext'
import { SiteNav } from '@/components/SiteNav'
import { Hero } from '@/components/hero/Hero'
import { Predictor } from '@/components/Predictor'
import { Sections } from '@/components/Sections'

export default function Page() {
  return (
    <ThresholdProvider>
      <FieldBackground />
      <SiteNav />
      <main>
      <Hero />
      <Predictor />
      <Sections />
    </main>
    </ThresholdProvider>
  )
}
