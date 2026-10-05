"use client"

import { Button } from "@/components/ui/button"

// "[R123]" or "[R12, R13]": a citation of one or more source responses.
const CITATION = /\[R(\d+(?:\s*,\s*R?\d+)*)\]/g

/** Text in which every citation is a button that opens the source response. */
export function CitedText({ text, onOpen }: { text: string; onOpen: (responseId: number) => void }) {
  const parts: React.ReactNode[] = []
  let last = 0
  for (const match of text.matchAll(CITATION)) {
    parts.push(text.slice(last, match.index))
    for (const id of match[1].match(/\d+/g) ?? []) {
      parts.push(
        <Button key={`${match.index}-${id}`} variant="link" size="xs" className="h-auto px-1 align-baseline" onClick={() => onOpen(Number(id))}>
          R{id}
        </Button>
      )
    }
    last = (match.index ?? 0) + match[0].length
  }
  parts.push(text.slice(last))
  return <p className="leading-relaxed whitespace-pre-wrap">{parts}</p>
}

export function CitationButton({ id, onOpen }: { id: number; onOpen: (responseId: number) => void }) {
  return (
    <Button variant="outline" size="xs" onClick={() => onOpen(id)}>
      R{id}
    </Button>
  )
}
