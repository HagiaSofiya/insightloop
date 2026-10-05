"use client"

import Link from "next/link"
import { useEffect, useState } from "react"

import { StatusBadge } from "@/components/status-badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { api, RunSummary } from "@/lib/api"
import { money, when } from "@/lib/format"

export function RunsTable() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    const load = () =>
      api<RunSummary[]>("/workflows")
        .then((rows) => active && (setRuns(rows), setError(null)))
        .catch((e) => active && setError(e.message))
    load()
    const timer = setInterval(load, 5000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Studies</CardTitle>
        <CardDescription>Every run, newest first. Click one to watch it or read its results.</CardDescription>
      </CardHeader>
      <CardContent>
        {error && <p className="text-sm text-destructive">Could not reach the API: {error}</p>}
        {!runs && !error && <Skeleton className="h-32 w-full" />}
        {runs && runs.length === 0 && (
          <Empty>
            <EmptyHeader>
              <EmptyTitle>No studies yet</EmptyTitle>
              <EmptyDescription>Start one with the form.</EmptyDescription>
            </EmptyHeader>
          </Empty>
        )}
        {runs && runs.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Run</TableHead>
                <TableHead>Brief</TableHead>
                <TableHead>Respondents</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Cost</TableHead>
                <TableHead>Started</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {runs.map((run) => (
                <TableRow key={run.id}>
                  <TableCell>
                    <Link href={`/runs/${run.id}`} className="font-medium underline-offset-4 hover:underline">
                      #{run.id}
                    </Link>
                  </TableCell>
                  <TableCell className="max-w-md truncate">{run.brief}</TableCell>
                  <TableCell>{run.respondent_count}</TableCell>
                  <TableCell>
                    <StatusBadge status={run.status} />
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{money(run.cost_usd)}</TableCell>
                  <TableCell className="text-muted-foreground">{when(run.created_at)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
