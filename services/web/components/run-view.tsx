"use client"

import { PauseIcon, PlayIcon, RotateCcwIcon, XIcon } from "lucide-react"
import { useParams } from "next/navigation"
import { useState } from "react"
import { toast } from "sonner"

import { CitedText } from "@/components/cited-text"
import { GuideReview } from "@/components/guide-review"
import { InsightsView } from "@/components/insights-view"
import { ReasoningPanel } from "@/components/reasoning-panel"
import { RunProgress } from "@/components/run-progress"
import { SourceSheet } from "@/components/source-sheet"
import { StatusBadge } from "@/components/status-badge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { post } from "@/lib/api"
import { money } from "@/lib/format"
import { useRun } from "@/lib/use-run"

export function RunView() {
  const runId = Number(useParams<{ id: string }>().id)
  const { run, error, events, reportText, progress, connected, refresh, reconnect } = useRun(runId)
  const [citation, setCitation] = useState<number | null>(null)

  async function act(action: "pause" | "resume" | "cancel" | "retry") {
    try {
      await post(`/workflows/${runId}/${action}`)
      if (action === "retry") reconnect()
      refresh()
    } catch (e) {
      toast.error(e instanceof Error ? e.message : `Could not ${action}`)
    }
  }

  if (error && !run) return <p className="text-destructive">Could not load run {runId}: {error}</p>
  if (!run) return <Skeleton className="h-64 w-full" />

  const step = (name: string) => run.steps.find((s) => s.name === name)
  const awaitingReview = run.status === "paused" && run.review_guide && !run.guide_approved
  const guideQuestions: string[] = step("guide")?.output?.questions ?? []
  const report = run.status === "done" ? run.report : reportText || run.report
  const streaming = run.status === "running" && step("synthesize")?.status === "running"
  const canPause = run.status === "running" || run.status === "pending"

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-3">
            Run #{run.id}
            <StatusBadge status={run.status} />
            <Badge variant={connected ? "secondary" : "outline"}>{connected ? "live" : "not streaming"}</Badge>
          </CardTitle>
          <CardDescription>{run.brief}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <dl className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
            <div className="flex gap-1.5"><dt className="text-muted-foreground">Respondents</dt><dd>{run.respondent_count}</dd></div>
            <div className="flex gap-1.5"><dt className="text-muted-foreground">Cost so far</dt><dd className="tabular-nums">{money(run.cost_usd)}</dd></div>
            <div className="flex gap-1.5"><dt className="text-muted-foreground">Model calls</dt><dd>{run.llm_calls}</dd></div>
            <div className="flex gap-1.5"><dt className="text-muted-foreground">Tokens</dt><dd className="tabular-nums">{run.tokens.toLocaleString()}</dd></div>
          </dl>
          <div className="ml-auto flex gap-2">
            {canPause && <Button variant="outline" size="sm" onClick={() => act("pause")}><PauseIcon data-icon="inline-start" />Pause</Button>}
            {run.status === "paused" && !awaitingReview && <Button variant="outline" size="sm" onClick={() => act("resume")}><PlayIcon data-icon="inline-start" />Resume</Button>}
            {run.status === "failed" && <Button variant="outline" size="sm" onClick={() => act("retry")}><RotateCcwIcon data-icon="inline-start" />Retry</Button>}
            {["pending", "running", "paused", "failed"].includes(run.status) && <Button variant="outline" size="sm" onClick={() => act("cancel")}><XIcon data-icon="inline-start" />Cancel</Button>}
          </div>
        </CardContent>
      </Card>

      {awaitingReview && guideQuestions.length > 0 && (
        <GuideReview runId={runId} initial={guideQuestions} onDone={() => { refresh() }} />
      )}

      {report && (
        <Card>
          <CardHeader>
            <CardTitle>Report</CardTitle>
            <CardDescription>
              {streaming ? "Writing now. The text may be replaced if the critic asks for another pass." : "Citations like R12 open the respondent's original answer."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CitedText text={report} onOpen={setCitation} />
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="progress">
        <TabsList>
          <TabsTrigger value="progress">Progress</TabsTrigger>
          <TabsTrigger value="guide">Guide</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
          <TabsTrigger value="reasoning">Show reasoning</TabsTrigger>
        </TabsList>
        <TabsContent value="progress" className="pt-4">
          <RunProgress run={run} events={events} progress={progress} />
        </TabsContent>
        <TabsContent value="guide" className="pt-4">
          <Card>
            <CardHeader>
              <CardTitle>Discussion guide</CardTitle>
              <CardDescription>The questions every respondent was asked{run.guide_approved ? ", as you approved them" : ""}.</CardDescription>
            </CardHeader>
            <CardContent>
              {guideQuestions.length === 0 ? (
                <p className="text-sm text-muted-foreground">The guide has not been written yet.</p>
              ) : (
                <ol className="flex list-decimal flex-col gap-2 pl-5">{guideQuestions.map((q) => <li key={q}>{q}</li>)}</ol>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="insights" className="pt-4">
          <InsightsView studyId={run.study_id} ready={step("themes")?.status === "done"} onOpen={setCitation} />
        </TabsContent>
        <TabsContent value="reasoning" className="pt-4">
          <ReasoningPanel synthesize={step("synthesize")} />
        </TabsContent>
      </Tabs>

      <SourceSheet responseId={citation} onClose={() => setCitation(null)} />
    </div>
  )
}
