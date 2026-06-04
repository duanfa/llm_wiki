import type { SourceWatchConfig } from "@/stores/wiki-store"
import { normalizeSourceWatchConfig } from "@/lib/source-watch-config"
import { isWebRuntime } from "@/lib/web-api"

export type FileChangeKind = "created" | "modified" | "deleted"
export type FileChangeStatus = "pending" | "processing" | "done" | "failed" | "superseded"

export interface FileChangeTask {
  id: string
  projectId: string
  path: string
  kind: FileChangeKind
  status: FileChangeStatus
  hashBefore?: string | null
  hashAfter?: string | null
  size?: number | null
  mtimeMs?: number | null
  createdAt: number
  updatedAt: number
  retryCount: number
  error?: string | null
  needsRerun: boolean
}

export interface FileChangeQueue {
  version: number
  tasks: FileChangeTask[]
}

export interface FileChangeRescanResult {
  queue: FileChangeQueue
  changedTasks: FileChangeTask[]
}

export interface FileSyncPayload {
  projectId: string
  tasks: FileChangeTask[]
}

function emptyRescanResult(): FileChangeRescanResult {
  return { queue: { version: 1, tasks: [] }, changedTasks: [] }
}

async function invokeTauri<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core")
  return invoke<T>(command, args)
}

export function startProjectFileWatcher(
  projectId: string,
  projectPath: string,
  sourceWatchConfig?: SourceWatchConfig,
): Promise<FileChangeRescanResult> {
  if (isWebRuntime()) return Promise.resolve(emptyRescanResult())
  return invokeTauri<FileChangeRescanResult>("start_project_file_watcher", {
    projectId,
    projectPath,
    sourceWatchConfig: normalizeSourceWatchConfig(sourceWatchConfig),
  })
}

export function stopProjectFileWatcher(): Promise<void> {
  if (isWebRuntime()) return Promise.resolve()
  return invokeTauri<void>("stop_project_file_watcher")
}

export function rescanProjectFiles(
  projectId: string,
  projectPath: string,
  sourceWatchConfig?: SourceWatchConfig,
): Promise<FileChangeRescanResult> {
  if (isWebRuntime()) return Promise.resolve(emptyRescanResult())
  return invokeTauri<FileChangeRescanResult>("rescan_project_files", {
    projectId,
    projectPath,
    sourceWatchConfig: normalizeSourceWatchConfig(sourceWatchConfig),
  })
}

export function getFileChangeQueue(projectPath: string): Promise<FileChangeQueue> {
  if (isWebRuntime()) return Promise.resolve(emptyRescanResult().queue)
  return invokeTauri<FileChangeQueue>("get_file_change_queue", { projectPath })
}

export function retryFileChangeTask(
  projectId: string,
  projectPath: string,
  taskId: string,
): Promise<FileChangeQueue> {
  if (isWebRuntime()) return Promise.resolve(emptyRescanResult().queue)
  return invokeTauri<FileChangeQueue>("retry_file_change_task", { projectId, projectPath, taskId })
}

export function ignoreFileChangeTask(
  projectId: string,
  projectPath: string,
  taskId: string,
): Promise<FileChangeQueue> {
  if (isWebRuntime()) return Promise.resolve(emptyRescanResult().queue)
  return invokeTauri<FileChangeQueue>("ignore_file_change_task", { projectId, projectPath, taskId })
}
