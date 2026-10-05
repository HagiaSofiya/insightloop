"use client"

import { PlusIcon, Trash2Icon } from "lucide-react"
import { useState } from "react"
import { toast } from "sonner"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Spinner } from "@/components/ui/spinner"
import { Textarea } from "@/components/ui/textarea"
import { post } from "@/lib/api"

const MAX_QUESTIONS = 10

/** The human-in-the-loop gate: the run is paused until the guide is approved, with any edits applied. */
export function GuideReview({ runId, initial, onDone }: { runId: number; initial: string[]; onDone: () => void }) {
  const [questions, setQuestions] = useState(initial)
  const [busy, setBusy] = useState(false)
  const filled = questions.filter((q) => q.trim())

  async function approve() {
    setBusy(true)
    try {
      await post(`/workflows/${runId}/guide`, { questions: filled })
      toast.success("Guide approved. Interviews are starting.")
      onDone()
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not approve the guide")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Review the discussion guide</CardTitle>
        <CardDescription>The run is paused. Edit, remove or add questions. Every respondent will be asked exactly these.</CardDescription>
      </CardHeader>
      <CardContent>
        <FieldGroup>
          {questions.map((question, index) => (
            <Field key={index}>
              <FieldLabel htmlFor={`question-${index}`}>Question {index + 1}</FieldLabel>
              <div className="flex gap-2">
                <Textarea
                  id={`question-${index}`}
                  rows={2}
                  value={question}
                  onChange={(e) => setQuestions(questions.map((q, i) => (i === index ? e.target.value : q)))}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label={`Remove question ${index + 1}`}
                  disabled={questions.length === 1}
                  onClick={() => setQuestions(questions.filter((_, i) => i !== index))}
                >
                  <Trash2Icon />
                </Button>
              </div>
            </Field>
          ))}
        </FieldGroup>
        {filled.length === 0 && (
          <Alert variant="destructive" className="mt-4">
            <AlertTitle>A guide needs at least one question</AlertTitle>
            <AlertDescription>Write a question before approving.</AlertDescription>
          </Alert>
        )}
      </CardContent>
      <CardFooter className="flex gap-2">
        <Button variant="outline" disabled={questions.length >= MAX_QUESTIONS} onClick={() => setQuestions([...questions, ""])}>
          <PlusIcon data-icon="inline-start" />
          Add question
        </Button>
        <Button disabled={busy || filled.length === 0} onClick={approve}>
          {busy && <Spinner data-icon="inline-start" />}
          Approve and continue
        </Button>
      </CardFooter>
    </Card>
  )
}
