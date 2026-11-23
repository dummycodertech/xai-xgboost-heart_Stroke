import type { Config } from 'tailwindcss'

// >>> Every colour resolves to a CSS variable, so light/dark/monitor-inset all flip from data-theme
const config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        canvas: 'var(--bg-canvas)',
        surface: 'var(--bg-surface)',
        hover: 'var(--bg-hover)',
        line: 'var(--border-subtle)',
        control: 'var(--border-control)',
        ink: 'var(--text-primary)',
        muted: 'var(--text-muted)',
        olive: 'var(--color-olive)',
        burgundy: 'var(--color-burgundy)',
        ochre: 'var(--color-ochre)',
        'olive-dim': 'var(--olive-dim)',
        'burgundy-dim': 'var(--burgundy-dim)',
        'ochre-dim': 'var(--ochre-dim)',
      },
      fontFamily: {
        sans: ['var(--font-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'ui-monospace', 'monospace'],
        display: ['var(--font-display)', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config
export default config
