"use client"

import { CheckCircle2Icon, CircleDashedIcon, CircleXIcon, LoaderCircleIcon } from "lucide-react"

import { StatusBadge } from "@/components/status-badge"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { RunDetail, Step } from "@/lib/api"
import { money } from "@/lib/format"
import type { RunEvent } from "@/lib/use-run"

export const STEP_LABELS: Record<string, string> = {
  guide: "Write discussion guide",
  review: "Review guide (you)",
  respondents: "Interview respondents",
  embed: "Embed answers",
  themes: "Find themes",
  synthesize: "Write report",
}

const ICONS = { done: CheckCircle2Icon, running: LoaderCircleIcon, failed: CircleXIcon, pending: CircleDashedIcon }

function describe(event: RunEvent): string {
  const d = event.data
  switch (event.type) {
    case "run.created": return `Run created (${d.respondents} respondents)`
    case "step.started": return `Started: ${STEP_LABELS[d.step ?? ""] ?? d.step}${(d.attempt ?? 1) > 1 ? ` (attempt ${d.attempt})` : ""}`
    case "step.finished": return `Finished: ${STEP_LABELS[d.step ?? ""] ?? d.step} (cost so far ${money(d.cost_usd ?? 0)})`
    case "step.failed": return `Failed: ${STEP_LABELS[d.step ?? ""] ?? d.step}: ${d.error}`
    case "respondent.saved": return `Respondent saved (${d.completed} of ${d.total})`
    case "report.reset": return "Report draft started"
    case "report.delta": return `Report text (+${d.text?.length ?? 0} characters)`
    case "review.requested": return "Waiting for you to review the guide"
    case "guide.approved": return "Guide approved"
    case "run.paused": return "Run paused"
    case "run.resumed": return "Run resumed"
    case "run.cancelled": return "Run cancelled"
    case "run.retried": return d.from_step ? `Retrying from ${d.from_step}` : "Retrying"
    case "run.done": return "Run finished"
    case "run.failed": return `Run failed: ${d.error}`
    default: return event.type
  }
}

function StepRow({ step, progress }: { step: Step; progress: { completed: number; total: number } | null }) {
  const Icon = ICONS[step.status]
  const saved = step.output?.completed ?? 0
  const live =
    step.name === "respondents" && step.status !== "pending"
      ? (progress ?? (step.output ? { completed: saved, total: saved + (step.output.failed_slots?.length ?? 0) } : null))
      : null
  return (
    <li className="flex flex-col gap-2 py-3">
      <div className="flex items-center gap-3">
        <Icon className={step.status === "running" ? "animate-spin text-muted-foreground" : "text-muted-foreground"} />
        <span className="flex-1 font-medium">{STEP_LABELS[step.name] ?? step.name}</span>
        {step.attempts > 1 && <span className="text-xs text-muted-foreground">attempt {step.attempts}</span>}
        <StatusBadge status={step.status} />
      </div>
      {live && live.total > 0 && (
        <div className="flex items-center gap-3 pl-8">
          <Progress value={(live.completed / live.total) * 100} />
          <span className="text-xs whitespace-nowrap text-muted-foreground tabular-nums">
            {live.completed} of {live.total}
          </span>
        </div>
      )}
      {(step.output?.failed_slots?.length ?? 0) > 0 && (
        <p className="pl-8 text-sm text-muted-foreground">
          {step.output?.failed_slots?.length} respondent(s) could not be generated and were skipped.
        </p>
      )}
      {step.error && (
        <Alert variant="destructive" className="ml-8 w-auto">
          <AlertTitle>This step failed</AlertTitle>
          <AlertDescription>{step.error}</AlertDescription>
        </Alert>
      )}
    </li>
  )
}

export function RunProgress({ run, events, progress }: { run: RunDetail; events: RunEvent[]; progress: { completed: number; total: number } | null }) {
  const feed = events.filter((e) => e.type !== "report.delta").slice(-60).reverse()
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Steps</CardTitle>
          <CardDescription>Each step is saved, so a crashed run resumes where it stopped.</CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="divide-y">
            {run.steps
              .filter((step) => step.name !== "review" || run.review_guide)
              .map((step) => (
                <StepRow key={step.name} step={step} progress={progress} />
              ))}
          </ul>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Live events</CardTitle>
          <CardDescription>What the server reports, newest first.</CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-80">
            <ol className="flex flex-col gap-2 pr-3 text-sm">
              {feed.map((event) => (
                <li key={event.id} className="flex gap-2">
                  <span className="w-12 shrink-0 text-xs text-muted-foreground tabular-nums">#{event.id}</span>
                  <span>{describe(event)}</span>
                </li>
              ))}
              {feed.length === 0 && <li className="text-muted-foreground">Waiting for events...</li>}
            </ol>
          </ScrollArea>
        </CardContent>
      </Card>
    </div>
  )
}
