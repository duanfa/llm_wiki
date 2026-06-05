import type { EmbeddingConfig, LlmConfig, MultimodalConfig } from "@/stores/wiki-store"
import { webApi } from "@/lib/web-api"

interface ServerModelConfigResponse {
  ok: boolean
  llmConfig?: Partial<LlmConfig>
  multimodalConfig?: Partial<MultimodalConfig>
  embeddingConfig?: Partial<EmbeddingConfig>
}

export async function loadServerModelConfig(): Promise<ServerModelConfigResponse | null> {
  try {
    return await webApi<ServerModelConfigResponse>("/api/v1/config/model")
  } catch (err) {
    console.warn("[model-config] failed to load server defaults:", err)
    return null
  }
}
