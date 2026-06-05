import type { ActivityItem } from "@/stores/activity-store"
import { useWikiStore } from "@/stores/wiki-store"
import { isWebRuntime, jsonBody, webApi } from "@/lib/web-api"

let timer: ReturnType<typeof setTimeout> | null = null
let latestItems: ActivityItem[] = []

export function publishActivitySnapshot(items: ActivityItem[], event = "snapshot"): void {
  if (!isWebRuntime()) return
  latestItems = items
  if (timer) clearTimeout(timer)
  timer = setTimeout(() => {
    timer = null
    const project = useWikiStore.getState().project
    if (!project) return
    webApi(`/api/v1/projects/${encodeURIComponent(project.id)}/activity`, {
      method: "POST",
      body: jsonBody({ event, items: latestItems }),
    }).catch((err) => {
      console.warn("[activity-sync] failed to publish activity snapshot:", err)
    })
  }, 200)
}
