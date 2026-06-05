import type { FileNode, WikiProject } from "@/types/wiki"
import { ensureProjectId, upsertProjectInfo } from "@/lib/project-identity"
import { isWebRuntime, jsonBody, webApi } from "@/lib/web-api"

/** Raw shape returned by the Rust commands — id is attached client-side. */
interface RawProject {
  id?: string
  name: string
  path: string
}

async function invokeTauri<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core")
  return invoke<T>(command, args)
}

export async function readFile(path: string): Promise<string> {
  if (isWebRuntime()) {
    const result = await webApi<{ content: string }>("/api/v1/fs/read", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.content
  }
  return invokeTauri<string>("read_file", { path })
}

export async function writeFile(path: string, contents: string): Promise<void> {
  if (isWebRuntime()) {
    await webApi<{ ok: boolean }>("/api/v1/fs/write", {
      method: "POST",
      body: jsonBody({ path, contents }),
    })
    return
  }
  return invokeTauri<void>("write_file", { path, contents })
}

export async function writeFileAtomic(path: string, contents: string): Promise<void> {
  if (isWebRuntime()) {
    await webApi<{ ok: boolean }>("/api/v1/fs/write-atomic", {
      method: "POST",
      body: jsonBody({ path, contents }),
    })
    return
  }
  return invokeTauri<void>("write_file_atomic", { path, contents })
}

export async function listDirectory(path: string): Promise<FileNode[]> {
  if (isWebRuntime()) {
    return webApi<FileNode[]>("/api/v1/fs/list", {
      method: "POST",
      body: jsonBody({ path }),
    })
  }
  return invokeTauri<FileNode[]>("list_directory", { path })
}

export async function copyFile(
  source: string,
  destination: string
): Promise<void> {
  if (isWebRuntime()) {
    await webApi<{ ok: boolean }>("/api/v1/fs/copy-file", {
      method: "POST",
      body: jsonBody({ source, destination }),
    })
    return
  }
  return invokeTauri("copy_file", { source, destination })
}

export async function copyDirectory(
  source: string,
  destination: string
): Promise<string[]> {
  if (isWebRuntime()) {
    return webApi<string[]>("/api/v1/fs/copy-directory", {
      method: "POST",
      body: jsonBody({ source, destination }),
    })
  }
  return invokeTauri<string[]>("copy_directory", { source, destination })
}

export async function preprocessFile(path: string): Promise<string> {
  if (isWebRuntime()) {
    const result = await webApi<{ content: string }>("/api/v1/fs/preprocess", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.content
  }
  return invokeTauri<string>("preprocess_file", { path })
}

export async function deleteFile(path: string): Promise<void> {
  if (isWebRuntime()) {
    await webApi<{ ok: boolean }>("/api/v1/fs/delete", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return
  }
  return invokeTauri("delete_file", { path })
}

export async function findRelatedWikiPages(
  projectPath: string,
  sourceName: string
): Promise<string[]> {
  return invokeTauri<string[]>("find_related_wiki_pages", { projectPath, sourceName })
}

export async function createDirectory(path: string): Promise<void> {
  if (isWebRuntime()) {
    await webApi<{ ok: boolean }>("/api/v1/fs/mkdir", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return
  }
  return invokeTauri<void>("create_directory", { path })
}

export async function fileExists(path: string): Promise<boolean> {
  if (isWebRuntime()) {
    const result = await webApi<{ exists: boolean }>("/api/v1/fs/exists", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.exists
  }
  return invokeTauri<boolean>("file_exists", { path })
}

export async function getFileModifiedTime(path: string): Promise<number> {
  if (isWebRuntime()) {
    const result = await webApi<{ modifiedTime: number }>("/api/v1/fs/stat", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.modifiedTime
  }
  return invokeTauri<number>("get_file_modified_time", { path })
}

export async function getFileSize(path: string): Promise<number> {
  if (isWebRuntime()) {
    const result = await webApi<{ size: number }>("/api/v1/fs/stat", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.size
  }
  return invokeTauri<number>("get_file_size", { path })
}

export async function getFileMd5(path: string): Promise<string> {
  if (isWebRuntime()) {
    const result = await webApi<{ md5: string }>("/api/v1/fs/md5", {
      method: "POST",
      body: jsonBody({ path }),
    })
    return result.md5
  }
  return invokeTauri<string>("get_file_md5", { path })
}

/** Mirror of `commands::fs::FileBase64` (Rust side). */
export interface FileBase64 {
  base64: string
  mimeType: string
}

/**
 * Read any file off disk as base64 + a guessed mime type. The
 * vision-caption pipeline uses this to pick up extracted images
 * without having to read them as UTF-8 strings (PNG bytes aren't
 * valid UTF-8 — `readFile` would corrupt them).
 */
export async function readFileAsBase64(path: string): Promise<FileBase64> {
  if (isWebRuntime()) {
    return webApi<FileBase64>("/api/v1/fs/base64", {
      method: "POST",
      body: jsonBody({ path }),
    })
  }
  return invokeTauri<FileBase64>("read_file_as_base64", { path })
}

export async function createProject(
  name: string,
  path: string,
  projectId?: string,
): Promise<WikiProject> {
  const raw = isWebRuntime()
    ? await webApi<RawProject>("/api/v1/projects/create", {
        method: "POST",
        body: jsonBody({ name, projectId }),
      })
    : await invokeTauri<RawProject>("create_project", { name, path })
  const id = raw.id ?? await ensureProjectId(raw.path)
  await upsertProjectInfo(id, raw.path, raw.name)
  return { id, name: raw.name, path: raw.path }
}

export async function openProject(path: string): Promise<WikiProject> {
  const raw = isWebRuntime()
    ? await webApi<RawProject>("/api/v1/projects/open", {
        method: "POST",
        body: jsonBody({ path }),
      })
    : await invokeTauri<RawProject>("open_project", { path })
  const id = await ensureProjectId(raw.path)
  await upsertProjectInfo(id, raw.path, raw.name)
  return { id, name: raw.name, path: raw.path }
}

export async function openProjectFolder(path: string): Promise<void> {
  if (isWebRuntime()) {
    window.open(path, "_blank", "noopener,noreferrer")
    return
  }
  return invokeTauri<void>("open_project_folder", { path })
}

export async function clipServerStatus(): Promise<string> {
  if (isWebRuntime()) return "unavailable"
  return invokeTauri<string>("clip_server_status")
}

export async function apiServerStatus(): Promise<string> {
  if (isWebRuntime()) {
    const result = await webApi<{ status: string }>("/api/v1/health")
    return result.status
  }
  return invokeTauri<string>("api_server_status")
}

export async function apiServerReloadConfig(): Promise<string> {
  if (isWebRuntime()) return "ok"
  return invokeTauri<string>("api_server_reload_config")
}
