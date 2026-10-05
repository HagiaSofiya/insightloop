"use client"

import { ChevronsUpDownIcon } from "lucide-react"
import { useState } from "react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Analyst, AgentRun, api, Step } from "@/lib/api"
import { money } from "@/lib/format"

function AnalystTrace({ analyst }: { analyst: Analyst }) {
  const [open, setOpen] = useState(false)
  const [run, setRun] = useState<AgentRun | null>(null)
  const [error, setError] = useState<string | null>(null)

  function toggle(next: boolean) {
    setOpen(next)
    if (next && !run) api<AgentRun>(`/agent-runs/${analyst.run_id}`).then(setRun).catch((e) => setError(e.message))
  }

  return (
    <Collapsible open={open} onOpenChange={toggle} className="rounded-lg border">
      <div className="flex items-center gap-3 p-3">
        <span className="flex-1 text-sm">{analyst.sub_question}</span>
        {analyst.verified === false && <Badge variant="destructive">unverified</Badge>}
        {analyst.verified === true && <Badge variant="secondary">verified</Badge>}
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm">
            <ChevronsUpDownIcon data-icon="inline-start" />
            {open ? "Hide steps" : "Show steps"}
          </Button>
        </CollapsibleTrigger>
      </div>
      <CollapsibleContent>
        <div className="flex flex-col gap-3 border-t p-3">
          {error && <p className="text-sm text-destructive">{error}</p>}
          {!run && !error && <Skeleton className="h-16 w-full" />}
          {run && (
            <>
              <p className="text-xs text-muted-foreground">
                Analyst run #{run.id}, {run.status}, {run.cost_usd !== null ? money(run.cost_usd) : "cost unknown"}
                {run.retried && ", retried once after a failed check"}
              </p>
              <ol className="flex flex-col gap-3">
                {run.steps.map((step) => (
                  <li key={step.number} className="flex flex-col gap-1 text-sm">
                    <div className="flex items-center gap-2">
                      <Badge variant={step.kind === "final" ? "default" : "outline"}>{step.kind === "final" ? "answer" : step.tool}</Badge>
                      <span className="text-xs text-muted-foreground">
                        {step.tokens} tokens, {step.latency_ms} ms
                      </span>
                      {step.is_error && <Badge variant="destructive">error fed back to the model</Badge>}
                    </div>
                    {step.arguments !== null && (
                      <pre className="overflow-x-auto rounded-md bg-muted p-2 text-xs">{JSON.stringify(step.arguments)}</pre>
                    )}
                    {step.result && <pre className="max-h-40 overflow-auto rounded-md bg-muted p-2 text-xs whitespace-pre-wrap">{step.result}</pre>}
                  </li>
                ))}
              </ol>
              {run.problems.length > 0 && (
                <Alert variant="destructive">
                  <AlertTitle>This analyst failed verification</AlertTitle>
                  <AlertDescription>{run.problems.map((p) => `${p.claim}: ${p.detail}`).join("; ")}</AlertDescription>
                </Alert>
              )}
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}

export function ReasoningPanel({ synthesize }: { synthesize: Step | undefined }) {
  const details = synthesize?.status === "done" ? synthesize.output : null
  if (!details) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No reasoning yet</EmptyTitle>
          <EmptyDescription>The plan, the analysts&apos; tool calls and the verifier result appear once the report is written.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }
  const verification = details.verification
  const analysts = details.analysts ?? []
  if (!verification) return null
  return (
    <div className="flex flex-col gap-6">
      <Alert variant={verification.verified ? "default" : "destructive"}>
        <AlertTitle>{verification.verified ? "Verified" : "Not verified"}</AlertTitle>
        <AlertDescription>
          {verification.quotes_checked} quotes and {verification.numbers_checked} numbers in the report were checked against the data.
          {verification.problems.map((p) => ` ${p.claim}: ${p.detail}.`).join("")}{" "}
          This check proves quotes and numbers exist. It cannot tell whether they support the claim they are attached to.
        </AlertDescription>
      </Alert>
      <Card>
        <CardHeader>
          <CardTitle>Plan</CardTitle>
          <CardDescription>The planner split the question into these parts, and one analyst took each.</CardDescription>
        </CardHeader>
        <CardContent>
          <ol className="list-decimal pl-5 text-sm">
            {(details.plan ?? []).map((q) => (
              <li key={q}>{q}</li>
            ))}
          </ol>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Analysts</CardTitle>
          <CardDescription>Open one to see every tool it called and what came back.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {analysts.map((a) => (
            <AnalystTrace key={a.run_id} analyst={a} />
          ))}
        </CardContent>
      </Card>
      {(details.critiques ?? []).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Critic</CardTitle>
            <CardDescription>A second reader checks the draft for gaps. One extra round is allowed.</CardDescription>
          </CardHeader>
          <CardContent className="text-sm">
            {details.critiques?.map((c, i) => (
              <p key={i}>{c.ok ? "No gaps found." : `Gaps found, so more research ran: ${c.gaps.join("; ")}`}</p>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
