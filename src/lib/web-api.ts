const DEFAULT_WEB_API_BASE_URL =
  typeof window === "undefined" ? "http://127.0.0.1:8000" : `${window.location.protocol}//${window.location.hostname}:8000`

export const WEB_API_BASE_URL =
  (import.meta.env.VITE_LLM_WIKI_API_BASE_URL as string | undefined)?.replace(/\/+$/, "") ||
  DEFAULT_WEB_API_BASE_URL

export function isWebRuntime(): boolean {
  const runtime = import.meta.env.VITE_LLM_WIKI_RUNTIME as string | undefined
  if (runtime === "web") return true
  if (runtime === "tauri") return false
  return typeof window !== "undefined" && !("__TAURI_INTERNALS__" in window)
}

export async function webApi<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const token = import.meta.env.VITE_LLM_WIKI_API_TOKEN as string | undefined
  const headers = new Headers(init.headers)
  if (!headers.has("Content-Type") && init.body !== undefined) {
    headers.set("Content-Type", "application/json")
  }
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`)
  }

  const response = await fetch(`${WEB_API_BASE_URL}${path}`, { ...init, headers })
  const contentType = response.headers.get("Content-Type") || ""
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const message =
      typeof payload === "object" && payload && "detail" in payload
        ? String(payload.detail)
        : typeof payload === "object" && payload && "error" in payload
          ? String(payload.error)
          : String(payload || response.statusText)
    throw new Error(message)
  }

  return payload as T
}

export function jsonBody(value: unknown): BodyInit {
  return JSON.stringify(value)
}
