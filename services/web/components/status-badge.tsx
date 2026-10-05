import { Badge } from "@/components/ui/badge"
import type { RunStatus, StepStatus } from "@/lib/api"

const VARIANTS: Record<RunStatus | StepStatus, "default" | "secondary" | "destructive" | "outline"> = {
  done: "default",
  running: "secondary",
  pending: "outline",
  paused: "outline",
  failed: "destructive",
  cancelled: "outline",
}

export function StatusBadge({ status }: { status: RunStatus | StepStatus }) {
  return <Badge variant={VARIANTS[status]}>{status}</Badge>
}
