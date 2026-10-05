"use client"

import { useEffect, useState } from "react"

import { CitationButton } from "@/components/cited-text"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, Insights } from "@/lib/api"
import { percent } from "@/lib/format"

const WEAK_SEPARATION = 0.25 // below this the themes overlap a lot: say so instead of implying sharp categories

export function InsightsView({ studyId, ready, onOpen }: { studyId: number | null; ready: boolean; onOpen: (id: number) => void }) {
  const [insights, setInsights] = useState<Insights | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (studyId === null || !ready) return
    let active = true
    api<Insights>(`/studies/${studyId}/insights`)
      .then((data) => active && setInsights(data))
      .catch((e) => active && setError(e.message))
    return () => {
      active = false
    }
  }, [studyId, ready])

  if (!ready || studyId === null) {
    return (
      <Empty>
        <EmptyHeader>
          <EmptyTitle>No insights yet</EmptyTitle>
          <EmptyDescription>Themes appear here when the run has found them.</EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }
  if (error) return <p className="text-sm text-destructive">{error}</p>
  if (!insights) return <Skeleton className="h-48 w-full" />

  return (
    <div className="flex flex-col gap-6">
      {insights.questions.map((question) => (
        <Card key={question.question_number}>
          <CardHeader>
            <CardTitle>
              Q{question.question_number}. {question.question}
            </CardTitle>
            <CardDescription className="flex items-center gap-2">
              {question.responses} responses
              {question.silhouette !== null && question.silhouette < WEAK_SEPARATION && (
                <Badge variant="outline">themes overlap (silhouette {question.silhouette.toFixed(2)})</Badge>
              )}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Theme</TableHead>
                  <TableHead className="w-48">Respondents</TableHead>
                  <TableHead>Closest answers</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {question.themes.map((theme) => (
                  <TableRow key={theme.label}>
                    <TableCell className="max-w-xs align-top whitespace-normal">
                      <div className="font-medium">{theme.label}</div>
                      <div className="text-sm text-muted-foreground">{theme.description}</div>
                    </TableCell>
                    <TableCell className="align-top">
                      <div className="flex items-center gap-2">
                        <Progress value={theme.share * 100} />
                        <span className="text-sm whitespace-nowrap tabular-nums">
                          {theme.size} ({percent(theme.share)})
                        </span>
                      </div>
                    </TableCell>
                    <TableCell className="align-top whitespace-normal">
                      <ul className="flex flex-col gap-2">
                        {theme.examples.map((example) => (
                          <li key={example.response_id} className="flex items-start gap-2 text-sm">
                            <CitationButton id={example.response_id} onOpen={onOpen} />
                            <span>{example.answer}</span>
                          </li>
                        ))}
                      </ul>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
