import { useState } from "react"
import { useTranslation } from "react-i18next"
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { FolderOpen } from "lucide-react"
import { createProject, writeFile, createDirectory } from "@/commands/fs"
import { getTemplate } from "@/lib/templates"
import { TemplatePicker } from "@/components/project/template-picker"
import type { WikiProject } from "@/types/wiki"
import { normalizePath } from "@/lib/path-utils"
import { OUTPUT_LANGUAGE_OPTIONS } from "@/lib/output-language-options"
import { useWikiStore, type OutputLanguage } from "@/stores/wiki-store"
import { saveOutputLanguage } from "@/lib/project-store"
import { isWebRuntime } from "@/lib/web-api"

type DirectoryHandleLike = { name: string }

async function pickWebDirectoryName(): Promise<string | null> {
  const maybeWindow = window as unknown as {
    showDirectoryPicker?: () => Promise<DirectoryHandleLike>
  }

  if (maybeWindow.showDirectoryPicker) {
    try {
      const handle = await maybeWindow.showDirectoryPicker()
      return handle.name || null
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return null
      throw err
    }
  }

  return new Promise((resolve) => {
    const input = document.createElement("input")
    input.type = "file"
    input.multiple = true
    input.setAttribute("webkitdirectory", "")
    input.setAttribute("directory", "")
    input.style.display = "none"

    input.onchange = () => {
      const first = input.files?.[0] as (File & { webkitRelativePath?: string }) | undefined
      document.body.removeChild(input)
      if (!first) {
        resolve(null)
        return
      }
      const root = first.webkitRelativePath?.split("/")[0]
      resolve(root || first.name || null)
    }

    input.oncancel = () => {
      document.body.removeChild(input)
      resolve(null)
    }

    document.body.appendChild(input)
    input.click()
  })
}

interface CreateProjectDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onCreated: (project: WikiProject) => void
}

export function CreateProjectDialog({ open: isOpen, onOpenChange, onCreated }: CreateProjectDialogProps) {
  const { t } = useTranslation()
  const webRuntime = isWebRuntime()
  const [name, setName] = useState("")
  const [path, setPath] = useState("")
  const [selectedTemplate, setSelectedTemplate] = useState("general")
  // Empty string = "user hasn't picked yet"; we validate this on
  // submit so a fresh project never starts in implicit auto-detect
  // mode. Once chosen, the value is one of OUTPUT_LANGUAGE_OPTIONS
  // (`auto` is a valid explicit choice — the user is then opting
  // INTO auto-detect rather than getting it by accident).
  const [language, setLanguage] = useState<string>("")
  const [error, setError] = useState("")
  const [creating, setCreating] = useState(false)
  const setOutputLanguage = useWikiStore((s) => s.setOutputLanguage)

  async function handleBrowse() {
    const selected = webRuntime
      ? await pickWebDirectoryName()
      : await import("@tauri-apps/plugin-dialog").then(({ open }) =>
          open({
            directory: true,
            multiple: false,
            title: t("project.browse"),
          }),
        )
    if (selected) {
      const nextPath = Array.isArray(selected) ? selected[0] : selected
      setPath(nextPath)
      if (webRuntime && !name.trim()) setName(nextPath)
    }
  }

  async function handleCreate() {
    if (!name.trim() || !path.trim()) {
      setError(t("project.errorNameRequired"))
      return
    }
    if (!language) {
      setError(t("project.errorLanguageRequired"))
      return
    }
    setCreating(true)
    setError("")
    try {
      const project = await createProject(
        name.trim(),
        path.trim(),
        webRuntime ? path.trim() : undefined,
      )
      const pp = normalizePath(project.path)

      const template = getTemplate(selectedTemplate)
      await writeFile(`${pp}/schema.md`, template.schema)
      await writeFile(`${pp}/purpose.md`, template.purpose)
      for (const dir of template.extraDirs) {
        await createDirectory(`${pp}/${dir}`)
      }

      // Persist the user's language choice. The store / disk
      // mirror is what the rest of the app reads via
      // `getOutputLanguage()` — without this write the choice
      // wouldn't survive past the dialog closing.
      const lang = language as OutputLanguage
      setOutputLanguage(lang)
      await saveOutputLanguage(lang, project.id)

      onCreated(project)
      onOpenChange(false)
      setName("")
      setPath("")
      setSelectedTemplate("general")
      setLanguage("")
    } catch (err) {
      setError(String(err))
    } finally {
      setCreating(false)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg max-h-[90vh] grid-rows-[auto_1fr_auto] overflow-hidden">
        <DialogHeader>
          <DialogTitle>{t("project.createTitle")}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-4 py-4 overflow-y-auto min-h-0">
          <div className="flex flex-col gap-2">
            <Label htmlFor="name">{t("project.name")}</Label>
            <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder={t("project.namePlaceholder")} />
          </div>
          <div className="flex flex-col gap-2">
            <Label>{t("project.template")}</Label>
            <TemplatePicker selected={selectedTemplate} onSelect={setSelectedTemplate} />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="language">
              {t("project.aiOutputLanguage")} <span className="text-destructive">{t("project.aiOutputLanguageRequired")}</span>
            </Label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full rounded-md border bg-background px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-ring"
            >
              <option value="" disabled>
                {t("project.pickLanguage")}
              </option>
              {/*
                * "auto" is intentionally filtered out at project
                * creation time. Auto-detect is a fine post-hoc
                * setting (Settings → Output) for users who later
                * decide they want it, but at create time we force
                * an explicit commitment so the project never starts
                * in the implicit-detect mode that was the source
                * of "wiki content showed up in a language I didn't
                * expect" surprises.
                */}
              {OUTPUT_LANGUAGE_OPTIONS.filter((l) => l.value !== "auto").map((l) => (
                <option key={l.value} value={l.value}>
                  {l.label}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              {t("project.aiOutputLanguageHint")}
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="path">{webRuntime ? t("project.projectId") : t("project.parentDir")}</Label>
            <div className="flex gap-2">
              <Input
                id="path"
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder={webRuntime ? t("project.projectIdPlaceholder") : t("project.parentDirPlaceholder")}
                className="flex-1"
              />
              {!webRuntime && (
                <Button variant="outline" size="icon" onClick={handleBrowse} type="button">
                  <FolderOpen className="h-4 w-4" />
                </Button>
              )}
            </div>
            {webRuntime && (
              <p className="text-xs text-muted-foreground">
                {t("project.projectIdHint")}
              </p>
            )}
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>{t("project.cancel")}</Button>
          <Button onClick={handleCreate} disabled={creating}>{creating ? t("project.creating") : t("project.create")}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
