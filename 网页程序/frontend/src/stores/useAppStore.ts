import { create } from 'zustand'
import { checkHealth } from '@/services/api'

interface AppState {
  serviceEnabled: boolean
  isMobile: boolean
  navTransparent: boolean

  checkServiceHealth: () => Promise<void>
  setMobile: (val: boolean) => void
  setNavTransparent: (val: boolean) => void
}

export const useAppStore = create<AppState>((set) => ({
  serviceEnabled: true,
  isMobile: false,
  navTransparent: true,

  checkServiceHealth: async () => {
    try {
      const res = await checkHealth()
      set({ serviceEnabled: res.status === 'ok' })
    } catch {
      set({ serviceEnabled: false })
    }
  },

  setMobile: (val) => set({ isMobile: val }),
  setNavTransparent: (val) => set({ navTransparent: val }),
}))
