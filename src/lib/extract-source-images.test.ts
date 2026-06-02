import { describe, expect, it } from "vitest"
import { buildImageMarkdownSection, findLocalMarkdownImageRefs } from "./extract-source-images"

describe("findLocalMarkdownImageRefs", () => {
  it("extracts Obsidian and markdown local image references", () => {
    const refs = findLocalMarkdownImageRefs(`
![[attachments/chart.png]]
![Figure](images/plot%201.jpg "title")
![Remote](https://example.com/a.png)
![[attachments/chart.png|400]]
`)
    expect(refs).toEqual(["attachments/chart.png", "images/plot 1.jpg"])
  })

  it("ignores non-image links and remote/data references", () => {
    const refs = findLocalMarkdownImageRefs(`
![Doc](notes/page.md)
![Data](data:image/png;base64,abc)
![[draft.txt]]
`)
    expect(refs).toEqual([])
  })
})

describe("buildImageMarkdownSection", () => {
  it("labels full-page screenshots for multi-image flows", () => {
    const section = buildImageMarkdownSection([
      {
        index: 1,
        mimeType: "image/png",
        kind: "pageScreenshot",
        page: 2,
        width: 1600,
        height: 2200,
        relPath: "media/source/page-2.png",
        absPath: "/proj/wiki/media/source/page-2.png",
        sha256: "hash",
      },
    ])

    expect(section).toContain("### Page 2")
    expect(section).toContain("Full-page screenshot for multi-image flow:")
    expect(section).toContain("![](media/source/page-2.png)")
  })
})
