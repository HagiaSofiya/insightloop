"use client"

import { useCallback, useEffect, useRef, useState } from "react"

import { API, api, RunDetail } from "@/lib/api"

export type EventData = {
  step?: string
  attempt?: number
  error?: string
  cost_usd?: number
  completed?: number
  total?: number
  text?: string
  respondents?: number
  from_step?: string | null
}
export type RunEvent = { id: number; type: string; data: EventData }

const TYPES = [
  "run.created", "step.started", "step.finished", "step.failed", "respondent.saved", "report.reset", "report.delta",
  "review.requested", "guide.approved", "run.paused", "run.resumed", "run.cancelled", "run.retried", "run.done", "run.failed",
]
const REFETCH_ON = new Set(TYPES.filter((t) => !t.startsWith("report."))) // includes respondent.saved, so the cost keeps up
const FINAL = new Set(["run.done", "run.failed", "run.cancelled"])
const MAX_EVENTS = 300

/**
 * One run: its saved state (fetched) plus its live event stream (SSE).
 * The browser's EventSource resends Last-Event-ID by itself when the connection drops. We also ignore any event id
 * we have already handled, so a replay (page reload, reopened stream) never applies an event twice.
 */
export function useRun(runId: number) {
  const [run, setRun] = useState<RunDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<RunEvent[]>([])
  const [reportText, setReportText] = useState("")
  const [progress, setProgress] = useState<{ completed: number; total: number } | null>(null)
  const [connected, setConnected] = useState(false)
  const [generation, setGeneration] = useState(0)
  const lastId = useRef(0)
  const status = useRef<RunDetail["status"] | null>(null)

  const refresh = useCallback(async () => {
    try {
      setRun(await api<RunDetail>(`/workflows/${runId}`))
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load the run")
    }
  }, [runId])

  useEffect(() => {
    let active = true
    api<RunDetail>(`/workflows/${runId}`)
      .then((detail) => active && setRun(detail))
      .catch((e) => active && setError(e instanceof Error ? e.message : "Could not load the run"))
    return () => {
      active = false
    }
  }, [runId])

  useEffect(() => {
    status.current = run?.status ?? null
  }, [run?.status])

  useEffect(() => {
    const source = new EventSource(`${API}/workflows/${runId}/events`)
    let timer: ReturnType<typeof setTimeout> | undefined
    const refreshSoon = () => {
      clearTimeout(timer)
      timer = setTimeout(refresh, 150) // several events often arrive together: fetch once
    }
    const handle = (type: string) => (message: MessageEvent) => {
      const id = Number(message.lastEventId)
      if (id <= lastId.current) return
      lastId.current = id
      const data = JSON.parse(message.data)
      setEvents((previous) => [...previous.slice(-(MAX_EVENTS - 1)), { id, type, data }])
      if (type === "report.reset") setReportText("")
      if (type === "report.delta") setReportText((text) => text + (data.text ?? ""))
      if (type === "respondent.saved") setProgress({ completed: data.completed ?? 0, total: data.total ?? 0 })
      if (REFETCH_ON.has(type)) refreshSoon()
      if (FINAL.has(type)) {
        source.close() // the server ends the stream here; without this the browser would reconnect forever
        setConnected(false)
      }
    }
    TYPES.forEach((type) => source.addEventListener(type, handle(type)))
    source.onopen = () => setConnected(true)
    source.onerror = () => {
      setConnected(false)
      if (status.current && ["done", "failed", "cancelled"].includes(status.current)) source.close()
    }
    return () => {
      clearTimeout(timer)
      source.close()
    }
  }, [runId, generation, refresh])

  /** Open the stream again, for example after retrying a finished run. Already-seen events are skipped. */
  const reconnect = useCallback(() => setGeneration((g) => g + 1), [])

  return { run, error, events, reportText, progress, connected, refresh, reconnect }
}
