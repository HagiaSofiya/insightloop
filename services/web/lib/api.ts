// The browser talks straight to the FastAPI backend (it allows this origin with CORS).
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export type StepStatus = "pending" | "running" | "done" | "failed"
export type RunStatus = "pending" | "running" | "paused" | "cancelled" | "failed" | "done"

export type Problem = { kind: string; claim: string; detail: string }
export type Analyst = { run_id: number; sub_question: string; status: string; verified: boolean | null }

/** What a finished step saved. Which fields exist depends on the step. */
export type StepOutput = {
  questions?: string[] // guide
  completed?: number // respondents
  failed_slots?: number[]
  plan?: string[] // synthesize: the data behind the reasoning panel
  analysts?: Analyst[]
  critiques?: { ok: boolean; gaps: string[] }[]
  verification?: { verified: boolean; quotes_checked: number; numbers_checked: number; problems: Problem[] }
}

export type Step = {
  name: string
  status: StepStatus
  attempts: number
  error: string | null
  output: StepOutput | null
}

export type RunSummary = {
  id: number
  brief: string
  respondent_count: number
  status: RunStatus
  study_id: number | null
  created_at: string
  cost_usd: number
}

export type RunDetail = RunSummary & {
  report: string | null
  error: string | null
  review_guide: boolean
  guide_approved: boolean
  tokens: number
  llm_calls: number
  steps: Step[]
}

export type Example = { response_id: number; age: number | null; gender: string | null; answer: string }
export type Theme = {
  kind: "theme" | "outlier"
  label: string
  description: string
  size: number
  share: number
  examples: Example[]
}
export type QuestionInsights = {
  question_number: number
  question: string | null
  responses: number
  silhouette: number | null
  themes: Theme[]
}
export type Insights = { study_id: number; brief: string; questions: QuestionInsights[] }

export type SourceResponse = {
  id: number
  question_number: number
  question: string
  answer: string
  study_id: number
  name: string | null
  age: number | null
  gender: string | null
  city: string | null
  occupation: string | null
}

export type AgentStep = {
  number: number
  kind: "tool" | "final"
  tool: string | null
  arguments: unknown
  result: string | null
  is_error: boolean
  tokens: number
  latency_ms: number
}
export type AgentRun = {
  id: number
  question: string
  status: string
  answer: string | null
  verified: boolean | null
  retried: boolean
  cost_usd: number | null
  problems: Problem[]
  steps: AgentStep[]
}

export class ApiError extends Error {}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
    cache: "no-store",
  })
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)
    } catch {}
    throw new ApiError(detail)
  }
  return response.json()
}

export const post = <T,>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) })
