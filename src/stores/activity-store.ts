import { create } from "zustand"
import { publishActivitySnapshot } from "@/lib/activity-sync"

export interface ActivityItem {
  id: string
  type: "ingest" | "lint" | "query"
  title: string
  status: "running" | "done" | "error"
  detail: string
  filesWritten: string[]
  createdAt: number
}

interface ActivityState {
  items: ActivityItem[]
  addItem: (item: Omit<ActivityItem, "id" | "createdAt">) => string
  updateItem: (id: string, updates: Partial<Pick<ActivityItem, "status" | "detail" | "filesWritten">>) => void
  appendDetail: (id: string, text: string) => void
  clearDone: () => void
}

let counter = 0

export const useActivityStore = create<ActivityState>((set, get) => ({
  items: [],

  addItem: (item) => {
    const id = `activity-${++counter}`
    set((state) => ({
      items: [
        { ...item, id, createdAt: Date.now() },
        ...state.items,
      ],
    }))
    publishActivitySnapshot(get().items, "add")
    return id
  },

  updateItem: (id, updates) =>
    set((state) => {
      const next = {
        items: state.items.map((item) =>
          item.id === id ? { ...item, ...updates } : item
        ),
      }
      queueMicrotask(() => publishActivitySnapshot(get().items, "update"))
      return next
    }),

  appendDetail: (id, text) =>
    set((state) => {
      const next = {
        items: state.items.map((item) =>
          item.id === id ? { ...item, detail: item.detail + text } : item
        ),
      }
      queueMicrotask(() => publishActivitySnapshot(get().items, "update"))
      return next
    }),

  clearDone: () =>
    set((state) => {
      const next = {
        items: state.items.filter((i) => i.status === "running"),
      }
      queueMicrotask(() => publishActivitySnapshot(get().items, "clear"))
      return next
    }),
}))
