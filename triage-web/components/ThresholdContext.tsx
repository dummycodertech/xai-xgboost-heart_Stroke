"use client"
import { createContext, useContext, useState, type ReactNode } from 'react'
// One shared alert-threshold choice for the predictor and the operating-point section. Ids match STRATEGIES and OPERATING_POINTS.
const Ctx = createContext<{ id: string; setId: (v: string) => void }>({ id: 'cost_optimal', setId: () => {} })
export function ThresholdProvider({ children }: { children: ReactNode }) {
  const [id, setId] = useState('cost_optimal')
  return <Ctx.Provider value={{ id, setId }}>{children}</Ctx.Provider>
}
export const useThreshold = () => useContext(Ctx)
