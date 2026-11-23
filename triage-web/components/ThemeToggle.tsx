'use client'
import { useEffect, useState } from 'react'

export function ThemeToggle() {
  const [dark, setDark] = useState(true)
  useEffect(() => {
    const isDark = document.documentElement.dataset.theme === 'dark' || localStorage.getItem('theme') !== 'light'
    setDark(isDark)
    if (isDark) document.documentElement.dataset.theme = 'dark'
    else delete document.documentElement.dataset.theme
  }, [])
  const flip = () => {
    const next = !dark
    setDark(next)
    if (next) {
      document.documentElement.dataset.theme = 'dark'
      try { localStorage.setItem('theme', 'dark') } catch {}
    } else {
      delete document.documentElement.dataset.theme
      try { localStorage.setItem('theme', 'light') } catch {}
    }
  }
  return (
    <button onClick={flip} aria-pressed={dark} aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'} className="h-9 rounded-full border border-line px-4 font-mono text-xs text-ink transition-colors hover:bg-hover">
      {dark ? 'Light' : 'Dark'}
    </button>
  )
}
