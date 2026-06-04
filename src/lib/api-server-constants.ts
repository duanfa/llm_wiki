import { isWebRuntime, WEB_API_BASE_URL } from "@/lib/web-api"

export const API_SERVER_PORT = 19828
export const API_SERVER_BASE_URL = isWebRuntime() ? WEB_API_BASE_URL : `http://127.0.0.1:${API_SERVER_PORT}`
export const API_SERVER_HEALTH_URL = `${API_SERVER_BASE_URL}/api/v1/health`
