"use client"

import { useEffect, useState } from "react"

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, SourceResponse } from "@/lib/api"

function SourceBody({ responseId }: { responseId: number }) {
  const [source, setSource] = useState<SourceResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    api<SourceResponse>(`/responses/${responseId}`)
      .then((row) => active && setSource(row))
      .catch((e) => active && setError(e.message))
    return () => {
      active = false
    }
  }, [responseId])

  return (
    <>
      <SheetHeader>
        <SheetTitle>Response R{responseId}</SheetTitle>
        <SheetDescription>The original answer behind this citation.</SheetDescription>
      </SheetHeader>
      <div className="flex flex-col gap-4 px-4">
          {error && <p className="text-sm text-destructive">{error}</p>}
          {!source && !error && <Skeleton className="h-24 w-full" />}
          {source && (
            <>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-muted-foreground">
                  Question {source.question_number}
                </span>
                <p className="font-medium">{source.question}</p>
              </div>
              <blockquote className="border-l-2 pl-4 leading-relaxed">{source.answer}</blockquote>
              <p className="text-sm text-muted-foreground">
                {[source.name, source.age && `${source.age}`, source.gender, source.city, source.occupation].filter(Boolean).join(", ")}
              </p>
            </>
          )}
      </div>
    </>
  )
}

/** The side panel behind a citation: who said it, to which question, in their own words. */
export function SourceSheet({ responseId, onClose }: { responseId: number | null; onClose: () => void }) {
  return (
    <Sheet open={responseId !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent>{responseId !== null && <SourceBody key={responseId} responseId={responseId} />}</SheetContent>
    </Sheet>
  )
}
