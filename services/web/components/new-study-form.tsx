"use client"

import { useRouter } from "next/navigation"
import { useState } from "react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Field, FieldDescription, FieldGroup, FieldLabel } from "@/components/ui/field"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { post } from "@/lib/api"

export function NewStudyForm() {
  const router = useRouter()
  const [brief, setBrief] = useState("")
  const [respondents, setRespondents] = useState(10)
  const [review, setReview] = useState(true)
  const [busy, setBusy] = useState(false)

  async function submit(event: React.FormEvent) {
    event.preventDefault()
    setBusy(true)
    try {
      const run = await post<{ id: number }>("/workflows", { brief, respondents, review_guide: review })
      router.push(`/runs/${run.id}`)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not start the study")
      setBusy(false)
    }
  }

  return (
    <Card>
      <form onSubmit={submit}>
        <CardHeader>
          <CardTitle>New study</CardTitle>
          <CardDescription>
            Describe what you want to learn. The app writes a discussion guide, interviews synthetic respondents and finds themes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="brief">Research brief</FieldLabel>
              <Textarea
                id="brief"
                required
                maxLength={1000}
                rows={4}
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="A meal-kit subscription for busy parents: why do customers cancel?"
              />
            </Field>
            <Field>
              <FieldLabel htmlFor="respondents">Respondents</FieldLabel>
              <Input
                id="respondents"
                type="number"
                min={10}
                max={50}
                value={respondents}
                onChange={(e) => setRespondents(Number(e.target.value))}
              />
              <FieldDescription>10 to 50. Theme finding needs at least 10.</FieldDescription>
            </Field>
            <Field orientation="horizontal">
              <Switch id="review" checked={review} onCheckedChange={setReview} />
              <FieldLabel htmlFor="review">Let me review the guide before interviews start</FieldLabel>
            </Field>
          </FieldGroup>
        </CardContent>
        <CardFooter>
          <Button type="submit" disabled={busy || brief.trim().length === 0}>
            {busy && <Spinner data-icon="inline-start" />}
            Start study
          </Button>
        </CardFooter>
      </form>
    </Card>
  )
}
